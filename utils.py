#!/usr/bin/env python3

#!pip install numpy PIL cairo ipywidgets gi IPython multipledispatch

from __future__ import annotations
from typing import List, Tuple, Dict
from multipledispatch import dispatch
import dataclasses
from dataclasses import dataclass
from collections import defaultdict
from copy import deepcopy
from copy import copy as shallowcopy
import numpy as np
import functools
import PIL
import cairo
import math
import gi
import io
import PIL.Image
import base64
import ipywidgets
from IPython.core.display import display, HTML

gi.require_version("Poppler", "0.18")
gi.require_foreign("cairo")
from gi.repository import Poppler, Gio

def pil_to_datauri(img: PIL.Image) -> str:
    #converts PIL image to datauri
    data = io.BytesIO()
    img.save(data, "PNG")
    data64 = base64.b64encode(data.getvalue())
    return 'data:image/png;base64,'+data64.decode()

def display_pil(imgs: PIL.Image | List[PIL.Image], /, max_width='100%', actually_display=True):
    src = '''<div style='display: flex; gap: 10px'>'''
    if type(imgs) != list:
        imgs = [imgs]
    for img in imgs:
        src += f'''<img style='border: 1px solid black; max-width: {max_width}' src='{pil_to_datauri(img)}'>'''
    src += '</div>'
    if actually_display:
        display(HTML(src))
    else:
        return HTML(src)
    
def cairo_ctx_to_pil_slow(ctx: cairo.Context) -> PIL.Image:
    buf = io.BytesIO()
    ctx.get_target().write_to_png(buf)
    return PIL.Image.open(buf)

def cairo_ctx_to_pil_fast(ctx: cairo.Context) -> PIL.Image:
    surface = ctx.get_target()
    # Don't ask me why the third to last parameter has to be "RGBA". I tried "RGB" but it didn't work.
    img = PIL.Image.frombuffer("RGB", (surface.get_width(), surface.get_height()), surface.get_data(), "raw", "RGBA", 0, 1)
    # Fix channel mixup
    R, G, B, A = img.split()
    img = PIL.Image.merge("RGB", (B, G, R))
    return img
    
def cairo_ctx_to_ipy_slow(ctx: cairo.Context) -> ipywidgets.Image:
    buf = io.BytesIO()
    ctx.get_target().write_to_png(buf)
    return ipywidgets.Image(value=buf.getvalue(), format='png')
    
    
def clone_cairo_context(ctx: cairo.Context) -> cairo.Context:
    surf = ctx.get_target()
    new_surf = surf.create_similar_image(cairo.Format.RGB24, surf.get_width(), surf.get_height())
    new_ctx = cairo.Context(new_surf)
    new_ctx.save()
    new_ctx.set_source_surface(surf)
    new_ctx.paint()
    new_ctx.restore()
    new_ctx.set_matrix(ctx.get_matrix())
    return new_ctx

@dataclass
class Point():
    x: float = None
    y: float = None

@dataclass
class Rect():
    left: float = None # x1
    bottom: float = None # y2
    right: float = None # x2
    top: float = None # y1
    
    @staticmethod
    def from_poppler(prect: Poppler.Rectangle):
        return Rect(prect.x1, prect.y2, prect.x2, prect.y1)
    
    def to_poppler(self) -> Poppler.Rectangle:
        prect = Poppler.Rectangle()
        prect.x1 = self.left
        prect.y1 = self.top
        prect.x2 = self.right
        prect.y2 = self.bottom
        return prect
    
    @property
    def center(self) -> Point:
        return Point((self.left+self.right)/2, (self.top+self.bottom)/2)
    
    @property
    def width(self) -> float:
        return self.right-self.left
    
    @property
    def height(self) -> float:
        return self.bottom-self.top
    
    # Manhathan distance
    def distance_to(self, other: rect, /, mode='max') -> float:
        Δhorizontal = abs(self.center.x - other.center.x) - (self.width + other.width)/2
        Δvertical = abs(self.center.y - other.center.y) - (self.height + other.height)/2
        if mode == 'max':
            return max(Δvertical, Δhorizontal)
        elif mode == 'min':
            return min(Δvertical, Δhorizontal)
        elif mode == 'horizontal':
            return Δhorizontal
        elif mode == 'vertical':
            return Δvertical
        elif mode == 'euclidean':
            return math.sqrt(Δvertical**2 + Δhorizontal**2)
        else:
            raise ValueError(f'Invalid mode={mode!r}')
    
    def union(self, other: Rect) -> Rect:
        left = min(self.left, other.left)
        right = max(self.right, other.right)
        bottom = max(self.bottom, other.bottom)
        top = min(self.top, other.top)
        return Rect(left, bottom, right, top)
    
    def __add__(self, other: Rect) -> Rect:
        return self.union(other)
    
    # this is likely buggy
    def intersection(self, other: Rect) -> Rect:
        left = max(self.left, other.left)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)
        top = max(self.top, other.top)
        if left > right:
            return None
        if top > bottom:
            return None
        return Rect(left, bottom, right, top)
    
    def __eq__(self, other: Rect) -> bool:
        return self.top == other.top and self.bottom == other.bottom and self.left == other.left and self.right == other.right
    
    def __hash__(self):
        return hash((self.top, self.bottom, self.left, self.right))
    
    def __repr__(self):
        return f'Rect(x: {self.center.x:.2f}, y: {self.center.y:.2f}, w: {self.width:5.2f}, h: {self.height:5.2f})'

@dataclass
class Color():
    red: int = None
    green: int = None
    blue: int = None
    alpha: int | None = None

    def hexcode(self) -> str:
        if self.alpha is None:
            return f'#{self.red:02}{self.green:02}{self.blue:02}'
        else:
            return f'#{self.red:02}{self.green:02}{self.blue:02}{self.alpha:02}'
    
    def __repr__(self):
        return f'Color({self.hexcode()})'
    
    def __hash__(self):
        return hash((self.red, self.green, self.blue, self.alpha))
    
    def __eq__(self, other: Color) -> bool:
        if self.alpha is None and other.alpha is not None:
            return False
        if self.alpha is not None and other.alpha is None:
            return False
        return self.red == other.red and self.green == other.green and self.blue == other.blue
    
    @staticmethod
    def from_poppler(color: Poppler.Color) -> Color:
        return Color(color.red, color.green, color.blue, None)
        
@dataclass
class Char():
    page: int = None
    font_size: float = None
    font_name: str = None
    color: Color = None
    underlined: bool = None
    text: str = None
    rect: Rect = None
    
    def __eq__(self, other: Char) -> bool:
        return self.page == other.page and \
                self.font_size == other.font_size and \
                self.font_name == other.font_name and \
                self.color == other.color and \
                self.underlined == other.underlined and \
                self.text == other.text and \
                self.rect == other.rect
    
    def __hash__(self):
        return hash((self.page, self.font_size, self.font_name, self.color, self.underlined, self.text, self.rect))

    def __repr__(self):
        return f'Char(text={self.text!r}, page={self.page}, font_size={self.font_size}, font_name={self.font_name!r}, color={self.color}, underlined={self.underlined}, rect={self.rect!r})'

class TextLine:
    pass
    
@dataclass
class TextLine():
    page: int = None
    chars: List[Char] = dataclasses.field(default_factory=list)
    
    @property
    def text(self) -> str:
        return ''.join(map(lambda char: char.text, self.chars))
    
    @property
    def rect(self) -> Rect:
        if len(self.chars) == 0:
            return None
        return sum(map(lambda char: char.rect, self.chars), self.chars[0].rect)
    
    @dispatch(Char)
    def __iadd__(self, other: Char):
        self.chars.append(other)
        return self
            
    @dispatch(Char)
    def __add__(self, other: Char) -> TextLine:
        self2 = shallowcopy(self)
        self2 += other
        return self2
    
    @dispatch(TextLine)
    def __add__(self, other: TextLine) -> TextBlock:
        self2 = deepcopy(self)
        self2 += other
        return TextBlock(self.page, [self, other])
        
    
@dataclass
class TextBlock():
    page: int = None
    lines: List[TextLine] = dataclasses.field(default_factory=list)
    
    @property
    def text(self) -> str:
        return ''.join(map(lambda line: line.text, self.lines))
    
    @property
    def rect(self) -> Rect:
        if len(self.lines) == 0:
            return None
        return sum(map(lambda line: line.rect, self.lines), self.lines[0].rect)
    
    @dispatch(TextLine)
    def __iadd__(self, other: TextLine):
        self.lines.append(other)
        return self
            
    @dispatch(TextLine)
    def __add__(self, other: TextLine) -> TextBlock:
        self2 = shallowcopy(self)
        self2 += other
        return self2
    
    @staticmethod
    def split_by_indent(blocks: List[TextBlock], /, min_indent: float = 25) -> List[TextBlock]:
        output = []
        for block in blocks:
            first_line = True
            for line in block.lines:
                indent = line.rect.left - block.rect.left
                new_paragraph = indent > min_indent
                if first_line or new_paragraph:
                    output.append(TextBlock(line.page, [line]))
                    first_line = False
                else:
                    output[-1].lines.append(line)
        return output
    
class PdfPage:
    page_num: int = None
    _poppler_page: Poppler.Page = None
    _cairo_surf: cairo.Surface = None
    _cairo_ctx: cairo.Context = None    
    _pil_img: PIL.Image = None
    _chars: List[Char] = None
    _lines: List[Char] = None
    _blocks: List[Char] = None
    _paragraphs: List[Char] = None
    
    def __init__(self, poppler_page: Poppler.Page, dpi=300) -> PdfPage:
        self._poppler_page = poppler_page
        
        scaling_factor = dpi/72 # 1 inch = 72 points and PDF is in points
        width_in_points, height_in_points = poppler_page.get_size()[0], poppler_page.get_size()[1]
        width_in_pixels, height_in_pixels = int(scaling_factor*width_in_points), int(scaling_factor*height_in_points)
        
        # Prepare our surface and context for rendering the PDF page
        surface = cairo.ImageSurface(cairo.Format.RGB24, width_in_pixels, height_in_pixels)
        ctx = cairo.Context(surface)
        ctx.scale(width_in_pixels/width_in_points, height_in_pixels/height_in_points)
        # Paint the background white to avoid transparency issues
        ctx.save()
        ctx.set_source_rgba(1, 1, 1, 1)
        ctx.paint()
        ctx.restore()
        # Actually render the PDF page
        poppler_page.render(ctx)
        
        # Don't ask me why that third to last argument has to be "RGBA" but when I tried "RGB" it failed
        pil_img = cairo_ctx_to_pil_fast(ctx)
        
        self._poppler_page = poppler_page
        self._cairo_surf = surface
        self._cairo_ctx = ctx
        self._pil_img = pil_img
        self.page_num = poppler_page.get_index()
        
    # Returns the image data for a rectange of the page
    def crop(self, rect: Rect) -> PIL.Image:
        (left, top) = self._cairo_ctx.user_to_device(rect.left, rect.top)
        (width, height) = self._cairo_ctx.user_to_device_distance(rect.width, rect.height)
        return self._pil_img.crop((left, top, left+width, top+height))
    
    def chars(self):
        if self._chars is not None:
            return self._chars
        # Get the pairs (char value, char position/bounding box)
        page_text = self._poppler_page.get_text()
        char_rects = self._poppler_page.get_text_layout()[1]
        
        # For each block of styling, apply to the previously obtained pairs (char, rect)
        output = []
        for attribute_block in self._poppler_page.get_text_attributes():
            color = Color.from_poppler(attribute_block.color)
            for i in range(attribute_block.start_index, attribute_block.end_index+1):
                output.append(Char(
                    self.page_num,
                    attribute_block.font_size,
                    attribute_block.font_name,
                    color,
                    attribute_block.is_underlined,
                    page_text[i],
                    Rect.from_poppler(char_rects[i])))
        self._chars = output
        return output
    
    def apply_char_modifier_function(self, function):
        self.chars() # Ensure that self._chars is populated
        for char in self._chars:
            function(char, self.crop(char.rect))
    
    def highlight_chars(self, from_char: int = 0, n_chars: int = None, ϵ: float = 0.1) -> PIL.Image:
        chars = self.chars()
        ctx = clone_cairo_context(self._cairo_ctx)
    
        # Ensure we don't overrun the available characters on a page
        to_char = len(chars)
        if n_chars is not None:
            to_char = min(to_char, from_char + n_chars)
    
        for char in chars[from_char:to_char]:
            rect = char.rect
            
            # Miniscule characters are represented by a dot
            if rect.width < ϵ and rect.height < ϵ:
                ctx.arc(rect.left, rect.top, 2, 0, 2 * math.pi)
                ctx.fill()
            else:
                # Draw a thin bounding rectange on each character
                ctx.rectangle(rect.left, rect.top, rect.width, rect.height)
                if char.underlined:
                    ctx.set_source_rgba(1.0, 0.0, 0.0, 1)
                else:
                    ctx.set_source_rgba(0.0, 0.0, 1.0, 0.5)
                ctx.set_line_width(0.2)
                ctx.stroke()
        img = cairo_ctx_to_pil_fast(ctx)
        img.thumbnail((1000, 1000), PIL.Image.LANCZOS)
        return img
    
    def highlight_things(self, things: list, /, rect_getter = None, color_getter = None) -> PIL.Image:
        ctx = clone_cairo_context(self._cairo_ctx)
    
        def get_thing_rect(thing):
            try:
                return thing.rect
            except:
                return thing['rect']
            
        def get_thing_color(thing):
            try:
                return thing.color
            except:
                return thing['color']
        
        if rect_getter is None:
            rect_getter = get_thing_rect
        if color_getter is None:
            color_getter = get_thing_color
            
        for thing in things:
            try:
                color = color_getter(thing)
                alpha = color.alpha if color.alpha is not None else 0.5
                ctx.set_source_rgba(color.red, color.green, color.blue, alpha)
            except:
                ctx.set_source_rgba(0.0, 0.0, 1.0, 0.5)
            try:
                rect = rect_getter(thing)
                ctx.rectangle(rect.left, rect.top, rect.width, rect.height)
                ctx.set_line_width(0.2)
                ctx.stroke()
            except:
                pass
        img = cairo_ctx_to_pil_fast(ctx)
        img.thumbnail((1000, 1000), PIL.Image.LANCZOS)
        return img
    
    def lines(self, /, max_horiz_dist: float = 1, max_vert_dist: float = -1) -> List[TextLine]:
        if self._lines is not None:
            return self._lines
        
        chars = self.chars()
        if len(chars) == 0:
            return []
        
        # This shit is way too slow. I need something to index my searches. Perhaps rtree
        
        lines = []
        free_chars = set(chars)
        for seed_char in chars:
            if seed_char not in free_chars:
                continue
            grouping, line = True, TextLine(self.page_num, [seed_char])
            char = seed_char
            free_chars.remove(seed_char)
            while grouping:
                best_match = None
                best_dist = float('inf')
                for char2 in free_chars:
                    if char == char2:
                        continue
                    dist_vert = char.rect.distance_to(char2.rect, mode='vertical')
                    # We require some overlap between the old char and the new one
                    if dist_vert > -1:
                        continue
                    horiz_dist = char.rect.distance_to(char2.rect, mode='horizontal')
                    if horiz_dist < best_dist and char != char2:
                        best_dist = horiz_dist
                        best_match = char2

                if best_dist < max_horiz_dist:
                    line += best_match
                    free_chars.remove(best_match)
                    char = best_match
                else:
                    grouping = False
                    lines.append(line)

        old_lines, lines = lines[1:], [lines[0]]
        for line in old_lines:
            old_line = lines[-1]
            # New lines are an annoying special case as the \n has no width nor height
            if line.text == '\n':
                old_line.chars += line.chars
                continue
            
            dist_vert = line.rect.distance_to(old_line.rect, mode='vertical')
            dist_horiz = line.rect.distance_to(old_line.rect, mode='horizontal')
            #print(f'{line.text!r} {old_line.text!r} {dist_vert} {dist_horiz}')
            if dist_vert <= -10:
                old_line.chars += line.chars
            else:
                lines.append(line)
            
        self._lines = lines        
        return lines
    
    def blocks(self, /, max_interline_dist: float = 5) -> List[TextBlock]:
        lines = self.lines()
        if len(lines) == 0:
            return []
        
        last_line = lines[0]
        blocks = [TextBlock(self.page_num, [last_line])]
        for line in lines[1:]:
            dist = last_line.rect.distance_to(line.rect)
            if dist < max_interline_dist:
                blocks[-1] += line
            else:
                blocks += [TextBlock(self.page_num, [line])]
            last_line = line
        return blocks

class PdfDocument:
    _poppler_document: Poppler.Document = None
    num_pages: int = None
    
    def __init__(self, filename: str) -> PdfDocument:
        with open(filename, 'rb') as fp:
            input_stream = Gio.MemoryInputStream.new_from_data(fp.read())
            self._poppler_document = Poppler.Document.new_from_stream(input_stream, -1, None, None)
        self.num_pages = self._poppler_document.get_n_pages()
            
    def get_page(self, page_num: int) -> PdfPage:
        return PdfPage(self._poppler_document.get_page(page_num-1))