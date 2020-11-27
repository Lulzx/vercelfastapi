# -*- coding: utf-8 -*-

import os
import re
import zipfile
from base64 import b64encode
from dataclasses import asdict
from html.parser import HTMLParser
from os.path import dirname, basename, join, splitext, abspath
from pathlib import Path
from typing import List
from urllib.parse import urlencode, parse_qs, urlparse

import requests
import fastapi.middleware.cors
from audiobooker.scrappers.librivox import Librivox
from bs4 import BeautifulSoup as bs
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from lxml import etree
from pydantic import AnyUrl, BaseModel
from pydantic.dataclasses import dataclass

parser = HTMLParser()


@dataclass
class LibGen:
    req: str
    column: str
    page: int


@dataclass
class GoogleBooks:
    q: str


@dataclass
class Result:
    id: int
    author: str
    title: str
    publisher: str
    year: str
    size: str
    extension: str
    download: AnyUrl


@dataclass
class SearchResult:
    results: list[Result]
    count: int


class BookInfo(BaseModel):
    title: str
    subtitle: str
    description: str
    year: str
    author: str
    image: str
    direct_url: AnyUrl


class AudiobookInfo(BaseModel):
    title: str
    description: str
    authors: str
    url: str
    streams: List[str]


def parse_url_args(url):
    query = parse_qs(urlparse(url).query)
    return {k: v[0] if v and len(v) == 1 else v for k, v in query.items()}


def sanitize(row):
    indices = 5, 6
    row = [i for j, i in enumerate(row[:10]) if j not in indices]
    row = [p.replace("'", "'").replace('"', '"') for p in row]
    size = row[-3].split(" ")
    val = size[0]
    ext = size[1]
    if ext == "Kb":
        val = float(val) / 1024
        ext = "Mb"
    size = "{:.2f} {}".format(round(float(val), 2), ext).replace(".00", "")
    row[-3] = size
    return row


def search(query: str = "", search_type: str = "title", page: int = 0):
    params = LibGen(req=query, column=search_type, page=page)
    query_string = urlencode(asdict(params), doseq=True)
    search_url = f"http://gen.lib.rus.ec/search.php?{query_string}"
    search_page = requests.get(search_url)
    soup = bs(search_page.text, "lxml")
    subheadings = soup.find_all("i")
    for subheading in subheadings:
        subheading.decompose()
    try:
        information_table = soup.find_all("table")[2]
        count = soup.find_all("font")[2].string.split("|")[0].split(" ")[0]
        raw_data = [
            [
                td.a["href"]
                if td.find("a")
                and td.find("a").has_attr("title")
                and td.find("a")["title"] != ""
                else "".join(td.stripped_strings)
                for td in row.find_all("td")
            ]
            for row in information_table.find_all("tr")[1:]
        ]
        cols = [
            "id",
            "author",
            "title",
            "publisher",
            "year",
            "size",
            "extension",
            "download",
        ]
        result = [dict(zip(cols, sanitize(row))) for row in raw_data]
    except:
        count = 0
        result = []
    return dict(results=result, count=count)


def extract_data(soup) -> dict:
    data = {}
    prefix = dict(author="Author", year="Year", description="Description")
    for key, value in prefix.items():
        tag = "div" if value == "Description" else "p"
        raw_data = soup.select_one(f"{tag}:contains({value})")
        data[key] = raw_data.get_text().split(
            ":", 1)[-1].strip() if raw_data else ""
    data.update(year=data["year"][-4:])
    return data


class Worker:
    def __init__(self, epub_path, output_dir):
        self.epub_path = epub_path
        script_dir = dirname(abspath(__file__))
        template_path = join(script_dir, "template.html")
        self.template = Path(template_path).read_text(encoding="utf-8")
        (epub_name_without_ext, _) = splitext(basename(self.epub_path))
        self.epub_name_without_ext = epub_name_without_ext
        self.output_dir = output_dir
        self.root_a_path = join(output_dir, epub_name_without_ext)
        self.unzip()
        opf_r_root_path = self.get_opf_r_root_path()
        self.index_a_path = join(self.root_a_path, "index.html")
        self.opf_a_path = join(self.root_a_path, opf_r_root_path)
        self.opf_a_dir = dirname(join(self.root_a_path, opf_r_root_path))
        self.ncx_r_opf_path, self.css_r_opf_path = self.paths_from_opf()
        self.ncx_a_path = join(self.opf_a_dir, self.ncx_r_opf_path)
        self.css_a_path = join(self.opf_a_dir, self.css_r_opf_path)
        self.already_gen_html = set()

    @staticmethod
    def get_xml_root(path):
        contents = Path(path).read_text(encoding="utf-8")
        contents = re.sub(' xmlns="[^"]+"', '', contents, count=1)
        contents = contents.encode("utf-8")
        root_path = etree.fromstring(contents)
        return root_path

    def get_opf_r_root_path(self):
        meta_a_path = (join(self.root_a_path, "META-INF/container.xml"))
        root_path = self.get_xml_root(meta_a_path)
        for item in root_path.findall(".//rootfiles/"):
            return item.attrib["full-path"]

    def read_xml(self, path):
        pass

    def paths_from_opf(self):
        ncx_r_opf_path = None
        css_r_opf_path = None
        root_path = self.get_xml_root(self.opf_a_path)
        for item in root_path.findall(".//manifest/"):
            href = item.attrib["href"]
            if "ncx" in item.attrib["media-type"]:
                ncx_r_opf_path = href
            if "css" in item.attrib["media-type"]:
                css_r_opf_path = href
        return ncx_r_opf_path, css_r_opf_path

    def get_index_loc(self):
        return self.index_a_path

    def _gen_menu_content(self, node, menus, contents, depth=0):
        for cc in node.findall("."):
            name = cc.find("./navLabel/text").text.strip()
            link = cc.find("./content")
            src = link.attrib["src"]
            unified_src = src
            no_hash_name = src
            if src.find('#') != -1:
                no_hash_name = src[:src.find("#")]
            if '#' not in src:
                unified_src = "#" + self.hash(src)
                anchor = f"<div id=\"{self.hash(src)}\"></div>"
                contents.append(anchor)
            else:
                unified_src = re.sub(r".+html", "", src)
            menus.append(f"<li><a href=\"{unified_src}\">{name}</a></li>")
            if no_hash_name in self.already_gen_html:
                continue
            self.already_gen_html.add(no_hash_name)
            washed_content = self.gen_content(
                join(dirname(self.ncx_a_path), no_hash_name))
            contents.append(washed_content)
            subs = cc.findall("./navPoint")
            if len(subs) > 0:
                for d in subs:
                    menus.append("<ul>")
                    self._gen_menu_content(d, menus, contents, depth + 1)
                    menus.append("</ul>")

    def gen_menu_content(self):
        menus = []
        contents = []
        root_path = self.get_xml_root(self.ncx_a_path)
        menus.append("<ul class=\"nav nav-sidebar \">")
        for c in root_path.findall("./navMap/navPoint"):
            self._gen_menu_content(c, menus, contents, 0)
        menus.append("</ul>")
        return "\n".join(menus), "".join(contents)

    def unzip(self):
        with zipfile.ZipFile(self.epub_path, 'r') as zip_ref:
            zip_ref.extractall(self.root_a_path)

    def gen_content(self, path):
        raw_text_content = Path(path).read_bytes()
        raw_content_dom = etree.HTML(raw_text_content)
        content = etree.tostring(raw_content_dom.xpath(
            "//body")[0], method='html').decode("utf-8")
        washed_content = self.wash_body(content)
        washed_content = self.wash_img_link(path, washed_content)
        return washed_content

    @staticmethod
    def wash_body(sub_content):
        tmp = sub_content.replace("<body", "<div")
        tmp = tmp.replace("</body>", "</div>")
        return tmp

    def wash_img_link(self, content_path, content):
        content = re.sub("(?<=src=\")(.*)(?=\")", lambda match: os.path.relpath(
            join(self.ncx_a_path, match.group(1))), content)
        return content

    @staticmethod
    def hash(s):
        import base64
        tag = base64.b64encode(s.encode('ascii'))
        tag = tag.decode("ascii")
        return tag.rstrip('=')

    def gen_r_css(self):
        with open(self.css_a_path, 'r') as file:
            return f'<style>{file.read()}</style>'

    def gen(self):
        menu, full_content = self.gen_menu_content()
        self.template = self.template.replace("${menu}$", menu)
        self.template = self.template.replace(
            "${title}$", self.epub_name_without_ext)
        self.template = self.template.replace("${content}$", full_content)
        self.template = self.template.replace("${css}$", self.gen_r_css())
        Path(join(self.output_dir, self.epub_name_without_ext, "./index.html")
             ).write_text(self.template, encoding="utf-8")


def replace_links(content):
    soup = bs(content, "lxml")
    for src in soup.findAll('a'):
        try:
            src["href"] = src["href"][src["href"].find("#"):]
        except KeyError:
            pass
    for img in soup.findAll('img'):
        img.attrs['loading'] = 'lazy'
    return soup


def processor(filepath):
    if filepath[0] != "." and filepath[0] != "/":
        filepath = "./" + filepath
    filepath = abspath(filepath)
    output_dir = "./static/"
    e = Worker(filepath, output_dir)
    e.gen()
    path = e.get_index_loc()
    with open(path, 'r') as markup:
        data = markup.read()
        data = replace_links(data)
    return str(data)


app = FastAPI()

app.add_middleware(
    fastapi.middleware.cors.CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return {"message": "Hello, World!"}


@app.get("/query/{search_type}/{query}/{page}", response_model=SearchResult)
def book_search(search_type: str, query: str, page: int):
    result = search(query, search_type, page)
    if not result["results"] and search_type == "title":
        params = GoogleBooks(q=query)
        query_string = urlencode(asdict(params), doseq=True)
        url = f"https://www.googleapis.com/books/v1/volumes?{query_string}"
        response = requests.get(url)
        data = response.json()
        if data["totalItems"] != 0:
            query = data["items"][0]["volumeInfo"]["title"]
            result = search(query, search_type, page)
    return result


@app.get("/book/{code}", response_model=BookInfo)
def book_info(code: str):
    base_url = "http://library.lol"
    link = f"{base_url}/main/{code}"
    response = requests.get(link)
    markup: str = response.text
    soup = bs(markup, "lxml")
    try:
        image = f"{base_url}{soup.find('img')['src']}"
        if not image:
            encoded_image_data = "NO_IMAGE"
        else:
            response = requests.get(image)
            encoded_image_data: str = (
                f"data:image/png;base64,{b64encode(response.content).decode('utf-8')}"
            )
            requests.aclose()
    except:
        encoded_image_data = "NO_IMAGE"
    try:
        direct_url = soup.select_one("a[href*=cloudflare]")["href"]
    except TypeError:
        direct_url = soup.select_one("a[href*=main]")["href"]
    except:
        direct_url = "/404"
    heading = soup.find("h1").text.split(":")
    title = heading[0]
    subtitle = heading[1].strip() if len(heading) > 1 else ""
    data = extract_data(soup)
    result = dict(
        title=title,
        subtitle=subtitle,
        **data,
        image=encoded_image_data,
        direct_url=direct_url,
    )
    return result


@app.get("/vox/{query}", response_model=AudiobookInfo)
def audiobook_search(query: str):
    book = Librivox.search_audiobooks(title=query)[0]
    authors = str(book.authors[0]).split(",")[0]
    result = dict(
        title=book.title,
        description=book.description,
        authors=authors,
        url=book.url,
        streams=book.streams,
    )
    return result


@app.get("/epub", response_class=HTMLResponse)
def epub(url: str):
    if url.endswith(".epub"):
        response = requests.get(url)
        try:
            filename = parse_url_args(url)["filename"]
        except:
            filename = url.split("/")[-1]
        with open(filename, 'wb') as f:
            f.write(response.content)
        return processor(filepath=filename)
    else:
        return "provide url to epub file."
