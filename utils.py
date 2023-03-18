from __future__ import annotations
from typing import List, Tuple, Dict
from dataclasses import dataclass
import numpy as np
import functools
import poppler
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
    
    def width(self) -> float:
        return self.right-self.left
    
    def height(self) -> float:
        return self.bottom-self.top
    
    def __repr__(self):
        return f'Rect(x: {self.left:.2f}, {self.right:.2f}, y: {self.top:.2f}, {self.bottom:.2f}, w: {self.width():5.2f}, h: {self.height():5.2f})'

@dataclass
class Color():
    red: int = None
    green: int = None
    blue: int = None

    def hexcode(self) -> str:
        return f'#{self.red:02}{self.green:02}{self.blue:02}'
    
    def __repr__(self):
        return f'Color({self.hexcode()})'
    
    @staticmethod
    def from_poppler(color: Poppler.Color) -> Color:
        return Color(color.red, color.green, color.blue)
        
@dataclass
class Char():
    page: int = None
    font_size: float = None
    font_name: str = None
    color: Color = None
    underlined: bool = None
    text: str = None
    rect: Rect = None

    def __repr__(self):
        return f'Char(text={self.text!r}, page={self.page}, font_size={self.font_size}, font_name={self.font_name!r}, color={self.color}, underlined={self.underlined}, rect={self.rect!r})'
    
class PdfPage:
    page_num: int = None
    _poppler_page: Poppler.Page = None
    _cairo_surf: cairo.Surface = None
    _cairo_ctx: cairo.Context = None    
    _pil_img: PIL.Image = None
    _chars: List[Char] = None
    
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
        (width, height) = self._cairo_ctx.user_to_device_distance(rect.width(), rect.height())
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
            if rect.width() < ϵ and rect.height() < ϵ:
                ctx.arc(rect.left, rect.top, 2, 0, 2 * math.pi)
                ctx.fill()
            else:
                # Draw a thin bounding rectange on each character
                ctx.rectangle(rect.left, rect.top, rect.width(), rect.height())
                if char.underlined:
                    ctx.set_source_rgba(1.0, 0.0, 0.0, 1)
                else:
                    ctx.set_source_rgba(0.0, 0.0, 1.0, 0.5)
                ctx.set_line_width(0.2)
                ctx.stroke()
        return cairo_ctx_to_pil_fast(ctx)

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