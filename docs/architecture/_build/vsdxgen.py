"""Minimal Visio VSDX writer for architecture diagrams.

Reuses the OPC skeleton of an existing reference drawing and regenerates
only the page content (shapes + connectors), the page sheet and docProps.
"""
from __future__ import annotations

import math
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

NS = "http://schemas.microsoft.com/office/visio/2012/main"

# ---------------------------------------------------------------- geometry ---
MARGIN = 0.55
COL_GAP = 1.35
ROW_GAP = 0.52
CHAR_W_CJK = 0.150
CHAR_W_ASCII = 0.079
LINE_H = 0.245
PAD_X = 0.22
PAD_Y = 0.17
BODY_SIZE = 0.145


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def text_width(text: str) -> float:
    w = 0.0
    for ch in text:
        w += CHAR_W_CJK if ord(ch) > 0x2E80 else CHAR_W_ASCII
    return w


@dataclass
class Node:
    nid: str
    label: str
    layer: int
    col: int
    fill: str = "#FDF0D5"
    line: str = "#A86B00"
    text_color: str = "#4A2D00"
    width: float | None = None
    height: float | None = None
    bold_first: bool = True
    font: str = "等线"
    shape: str = "rect"  # rect | round | stadium
    # filled by layout
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0

    def measure(self) -> None:
        lines = self.label.split("\n")
        n = len(lines)
        if self.width is None:
            longest = max(lines, key=text_width)
            self.w = max(2.35, min(6.6, text_width(longest) + 2 * PAD_X + 0.06))
        else:
            self.w = self.width
        self.h = self.height if self.height is not None else n * LINE_H + 2 * PAD_Y + 0.06


@dataclass
class Edge:
    src: str
    dst: str
    label: str = ""
    dashed: bool = False
    color: str = "#526477"
    arrow: bool = True
    width: float = 0.012
    label_pos: float = 0.5
    # routing: auto = vertical if layers differ, else horizontal
    route: str = "auto"


class Diagram:
    def __init__(self, title: str, subtitle: str = "", page_title: str = "", filename: str = ""):
        self.title = title
        self.subtitle = subtitle
        self.page_title = page_title or title
        self.filename = filename or self.page_title
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []

    def add(self, *nodes: Node) -> None:
        for nd in nodes:
            if nd.nid in self.nodes:
                raise ValueError(f"duplicate node id {nd.nid}")
            self.nodes[nd.nid] = nd

    def link(self, src: str, dst: str, **kw) -> None:
        self.edges.append(Edge(src, dst, **kw))

    # ------------------------------------------------------------- layout ---
    def layout(self) -> None:
        if getattr(self, "_laid_out", False):
            return
        for nd in self.nodes.values():
            nd.measure()

        rows: dict[int, list[Node]] = {}
        for nd in self.nodes.values():
            rows.setdefault(nd.layer, []).append(nd)

        # column widths per column index (global alignment); col=-1 means centred
        col_w: dict[int, float] = {}
        for nd in self.nodes.values():
            if nd.col < 0:
                continue
            col_w[nd.col] = max(col_w.get(nd.col, 0.0), nd.w)

        for layer in sorted(rows):
            for nd in rows[layer]:
                if nd.col >= 0:
                    nd.w = col_w[nd.col]

        y_cursor = MARGIN + 1.15  # leave room for header
        for layer in sorted(rows):
            rowh = max(nd.h for nd in rows[layer])
            for nd in rows[layer]:
                nd.y = y_cursor + (rowh - nd.h) / 2.0
            y_cursor += rowh + ROW_GAP

        x_cursor: dict[int, float] = {}
        x = MARGIN
        for col in sorted(col_w):
            x_cursor[col] = x
            x += col_w[col] + COL_GAP
        content_right = x - COL_GAP
        for nd in self.nodes.values():
            if nd.col >= 0:
                nd.x = x_cursor[nd.col]
            else:
                nd.x = (content_right + MARGIN - nd.w) / 2.0

        self.content_w = content_right + MARGIN
        self.content_h = y_cursor - ROW_GAP + MARGIN
        self._laid_out = True

    def page_size(self) -> tuple[float, float]:
        self.layout()
        # Visio page size in inches; cap to printable-ish but allow big.
        w = max(11.0, min(34.0, self.content_w + 1.0))
        h = max(8.5, min(44.0, self.content_h + 1.4))
        return w, h


# ------------------------------------------------------------------ writer ---
def _shape_xml(nd: Node, page_h: float, shape_id: int) -> str:
    pin_x = nd.x + nd.w / 2.0
    pin_y = page_h - (nd.y + nd.h / 2.0)
    radius = 0.10 if nd.shape in ("round", "stadium") else 0.0
    geom: list[str] = []

    if radius and min(nd.w, nd.h) > 3 * radius:
        r = radius
        w, h = nd.w, nd.h
        geom = [
            f'<Row T="MoveTo" IX="1"><Cell N="X" V="{r}"/><Cell N="Y" V="0"/></Row>',
            f'<Row T="LineTo" IX="2"><Cell N="X" V="{w-r}"/><Cell N="Y" V="0"/></Row>',
            f'<Row T="ArcTo" IX="3"><Cell N="X" V="{w}"/><Cell N="Y" V="{r}"/><Cell N="A" V="{r}"/></Row>',
            f'<Row T="LineTo" IX="4"><Cell N="X" V="{w}"/><Cell N="Y" V="{h-r}"/></Row>',
            f'<Row T="ArcTo" IX="5"><Cell N="X" V="{w-r}"/><Cell N="Y" V="{h}"/><Cell N="A" V="{r}"/></Row>',
            f'<Row T="LineTo" IX="6"><Cell N="X" V="{r}"/><Cell N="Y" V="{h}"/></Row>',
            f'<Row T="ArcTo" IX="7"><Cell N="X" V="0"/><Cell N="Y" V="{h-r}"/><Cell N="A" V="{r}"/></Row>',
            f'<Row T="LineTo" IX="8"><Cell N="X" V="0"/><Cell N="Y" V="{r}"/></Row>',
            f'<Row T="ArcTo" IX="9"><Cell N="X" V="{r}"/><Cell N="Y" V="0"/><Cell N="A" V="{r}"/></Row>',
            '<Row T="LineTo" IX="10" Del="1"/>',
        ]
    else:
        geom = [
            f'<Row T="RelMoveTo" IX="1"><Cell N="X" V="0"/><Cell N="Y" V="0"/></Row>',
            f'<Row T="RelLineTo" IX="2"><Cell N="X" V="1"/><Cell N="Y" V="0"/></Row>',
            f'<Row T="RelLineTo" IX="3"><Cell N="X" V="1"/><Cell N="Y" V="1"/></Row>',
            f'<Row T="RelLineTo" IX="4"><Cell N="X" V="0"/><Cell N="Y" V="1"/></Row>',
            f'<Row T="RelLineTo" IX="5"><Cell N="X" V="0"/><Cell N="Y" V="0"/></Row>',
        ]

    # first line bold-ish (larger), remaining lines smaller
    lines = nd.label.split("\n")
    cell_text = esc(lines[0])
    if len(lines) > 1:
        cell_text += "\n" + "\n".join(esc(x) for x in lines[1:])

    return (
        f'<Shape ID="{shape_id}" Type="Shape" LineStyle="3" FillStyle="3" TextStyle="3">'
        f'<Cell N="PinX" V="{pin_x:.4f}"/>'
        f'<Cell N="PinY" V="{pin_y:.4f}"/>'
        f'<Cell N="Width" V="{nd.w:.4f}"/>'
        f'<Cell N="Height" V="{nd.h:.4f}"/>'
        f'<Cell N="LocPinX" V="{nd.w/2:.4f}" F="Width*0.5"/>'
        f'<Cell N="LocPinY" V="{nd.h/2:.4f}" F="Height*0.5"/>'
        f'<Cell N="Angle" V="0"/>'
        f'<Cell N="FlipX" V="0"/>'
        f'<Cell N="FlipY" V="0"/>'
        f'<Cell N="ResizeMode" V="0"/>'
        f'<Cell N="ObjType" V="1"/>'
        f'<Cell N="FillForegnd" V="{nd.fill}"/>'
        f'<Cell N="LineColor" V="{nd.line}"/>'
        f'<Cell N="LineWeight" V="0.014"/>'
        f'<Cell N="VerticalAlign" V="1"/>'
        f'<Cell N="LeftMargin" V="0.10"/>'
        f'<Cell N="RightMargin" V="0.10"/>'
        f'<Cell N="TopMargin" V="0.06"/>'
        f'<Cell N="BottomMargin" V="0.06"/>'
        f'<Section N="Character">'
        f'<Row IX="0"><Cell N="Color" V="{nd.text_color}"/><Cell N="Size" V="{BODY_SIZE}"/><Cell N="Font" V="0"/></Row>'
        f'</Section>'
        f'<Section N="Geometry" IX="0"><Cell N="NoFill" V="0"/><Cell N="NoLine" V="0"/>'
        f'<Cell N="NoShow" V="0"/><Cell N="NoSnap" V="0"/><Cell N="NoQuickDrag" V="0"/>'
        + "".join(geom)
        + "</Section>"
        f"<Text><cp IX=\"0\"/>{cell_text}</Text>"
        "</Shape>"
    )


def _connector_xml(shape_id: int, x1: float, y1: float, x2: float, y2: float,
                   page_h: float, edge: Edge, src_id: int, dst_id: int,
                   label: str = "") -> str:
    bx, by = x1, page_h - y1
    ex, ey = x2, page_h - y2
    w = ex - bx
    h = ey - by
    pin_x = (bx + ex) / 2.0
    pin_y = (by + ey) / 2.0
    dash = '<Cell N="LinePattern" V="2"/>' if edge.dashed else ""
    arrow = f'<Cell N="EndArrow" V="13"/>' if edge.arrow else '<Cell N="EndArrow" V="0"/>'
    label = esc(label)
    tw = min(2.4, max(0.7, text_width(label) + 0.18))
    return (
        f'<Shape ID="{shape_id}" NameU="Dynamic connector" Name="Dynamic connector" Type="Shape" Master="2">'
        f'<Cell N="PinX" V="{pin_x:.4f}" F="Inh"/>'
        f'<Cell N="PinY" V="{pin_y:.4f}" F="Inh"/>'
        f'<Cell N="Width" V="{w:.4f}" F="GUARD(EndX-BeginX)"/>'
        f'<Cell N="Height" V="{h:.4f}" F="GUARD(EndY-BeginY)"/>'
        f'<Cell N="LocPinX" V="{w/2:.4f}" F="Inh"/>'
        f'<Cell N="LocPinY" V="{h/2:.4f}" F="Inh"/>'
        f'<Cell N="BeginX" V="{bx:.4f}" F="_WALKGLUE(BegTrigger,EndTrigger,WalkPreference)"/>'
        f'<Cell N="BeginY" V="{by:.4f}" F="_WALKGLUE(BegTrigger,EndTrigger,WalkPreference)"/>'
        f'<Cell N="EndX" V="{ex:.4f}" F="_WALKGLUE(EndTrigger,BegTrigger,WalkPreference)"/>'
        f'<Cell N="EndY" V="{ey:.4f}" F="_WALKGLUE(EndTrigger,BegTrigger,WalkPreference)"/>'
        f'<Cell N="LayerMember" V="0"/>'
        f'<Cell N="BegTrigger" V="2" F="_XFTRIGGER(Sheet{src_id}!EventXFMod)"/>'
        f'<Cell N="EndTrigger" V="2" F="_XFTRIGGER(Sheet{dst_id}!EventXFMod)"/>'
        f'<Cell N="ShapeRouteStyle" V="1"/>'
        f'<Cell N="ConFixedCode" V="6"/>'
        f'<Cell N="ConLineRouteExt" V="0"/>'
        f'<Cell N="TxtPinX" V="{w/2:.4f}"/>'
        f'<Cell N="TxtPinY" V="{(h/2 - (0.30 if abs(h) < 0.35 and abs(w) > 0.5 else 0.0)):.4f}"/>'
        f'<Cell N="TxtWidth" V="{tw:.4f}"/>'
        f'<Cell N="TxtHeight" V="0.18"/>'
        f'<Cell N="TxtLocPinX" V="{tw/2:.4f}"/>'
        f'<Cell N="TxtLocPinY" V="0.09"/>'
        f'<Cell N="GlueType" V="2"/>'
        f'<Cell N="ObjType" V="2"/>'
        f'<Cell N="LineColor" V="{edge.color}"/>'
        f'<Cell N="LineWeight" V="{edge.width}"/>'
        f'{dash}{arrow}'
        f'<Section N="Character"><Row IX="0"><Cell N="Color" V="#334155"/><Cell N="Size" V="0.10"/><Cell N="Font" V="0"/></Row></Section>'
        f'<Section N="Geometry" IX="0">'
        f'<Row T="MoveTo" IX="1"><Cell N="X" V="0"/><Cell N="Y" V="0"/></Row>'
        f'<Row T="LineTo" IX="2"><Cell N="X" V="{w:.4f}"/><Cell N="Y" V="{h:.4f}"/></Row>'
        f'<Row T="LineTo" IX="3" Del="1"/>'
        f"</Section>"
        f"<Text><cp IX=\"0\"/>{label}</Text>"
        "</Shape>"
    )


def _polyline_xml(shape_id: int, pts: list[tuple[float, float]], page_h: float,
                  edge: Edge) -> str:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    w = max(maxx - minx, 0.001)
    h = max(maxy - miny, 0.001)
    pin_x = minx + w / 2.0
    pin_y = page_h - (miny + h / 2.0)
    rows = []
    for i, (x, y) in enumerate(pts):
        lx = x - minx
        ly = (miny + h) - y
        tag = "MoveTo" if i == 0 else "LineTo"
        rows.append(f'<Row T="{tag}" IX="{i+1}"><Cell N="X" V="{lx:.4f}"/><Cell N="Y" V="{ly:.4f}"/></Row>')
    rows.append(f'<Row T="LineTo" IX="{len(pts)+1}" Del="1"/>')
    dash = '<Cell N="LinePattern" V="2"/>' if edge.dashed else ""
    arrow = '<Cell N="EndArrow" V="13"/>' if edge.arrow else ""
    return (
        f'<Shape ID="{shape_id}" Type="Shape" LineStyle="3" FillStyle="3" TextStyle="3">'
        f'<Cell N="PinX" V="{pin_x:.4f}"/>'
        f'<Cell N="PinY" V="{pin_y:.4f}"/>'
        f'<Cell N="Width" V="{w:.4f}"/>'
        f'<Cell N="Height" V="{h:.4f}"/>'
        f'<Cell N="LocPinX" V="{w/2:.4f}" F="Width*0.5"/>'
        f'<Cell N="LocPinY" V="{h/2:.4f}" F="Height*0.5"/>'
        f'<Cell N="Angle" V="0"/>'
        f'<Cell N="FlipX" V="0"/><Cell N="FlipY" V="0"/>'
        f'<Cell N="ObjType" V="1"/>'
        f'<Cell N="LineColor" V="{edge.color}"/>'
        f'<Cell N="LineWeight" V="{edge.width}"/>'
        f'<Cell N="FillForegnd" V="#FFFFFF"/>'
        f'{dash}{arrow}'
        f'<Section N="Geometry" IX="0"><Cell N="NoFill" V="1"/><Cell N="NoLine" V="0"/>'
        f'<Cell N="NoShow" V="0"/><Cell N="NoSnap" V="1"/><Cell N="NoQuickDrag" V="0"/>'
        + "".join(rows)
        + "</Section><Text><cp IX=\"0\"/></Text></Shape>"
    )


def _label_xml(shape_id: int, text: str, x: float, top_y: float, page_h: float,
               color: str = "#6A1B9A", align: str = "left") -> str:
    tw = text_width(text) + 0.14
    th = 0.22
    pin_x = (x + tw / 2.0) if align == "left" else x
    pin_y = page_h - (top_y + th / 2.0)
    return (
        f'<Shape ID="{shape_id}" Type="Shape" LineStyle="3" FillStyle="3" TextStyle="3">'
        f'<Cell N="PinX" V="{pin_x:.4f}"/>'
        f'<Cell N="PinY" V="{pin_y:.4f}"/>'
        f'<Cell N="Width" V="{tw:.4f}"/>'
        f'<Cell N="Height" V="{th:.4f}"/>'
        f'<Cell N="LocPinX" V="{tw/2:.4f}" F="Width*0.5"/>'
        f'<Cell N="LocPinY" V="{th/2:.4f}" F="Height*0.5"/>'
        f'<Cell N="Angle" V="0"/><Cell N="FlipX" V="0"/><Cell N="FlipY" V="0"/>'
        f'<Cell N="ObjType" V="1"/>'
        f'<Cell N="FillForegnd" V="#FFFFFF"/>'
        f'<Cell N="LineColor" V="#FFFFFF"/>'
        f'<Cell N="LineWeight" V="0.001"/>'
        f'<Section N="Character"><Row IX="0"><Cell N="Color" V="{color}"/><Cell N="Size" V="0.105"/><Cell N="Font" V="0"/></Row></Section>'
        f'<Section N="Geometry" IX="0"><Cell N="NoFill" V="1"/><Cell N="NoLine" V="1"/>'
        f'<Cell N="NoShow" V="0"/><Cell N="NoSnap" V="1"/><Cell N="NoQuickDrag" V="0"/>'
        f'<Row T="RelMoveTo" IX="1"><Cell N="X" V="0"/><Cell N="Y" V="0"/></Row>'
        f'<Row T="RelLineTo" IX="2"><Cell N="X" V="1"/><Cell N="Y" V="0"/></Row>'
        f'<Row T="RelLineTo" IX="3"><Cell N="X" V="1"/><Cell N="Y" V="1"/></Row>'
        f'<Row T="RelLineTo" IX="4"><Cell N="X" V="0"/><Cell N="Y" V="1"/></Row>'
        f'<Row T="RelLineTo" IX="5"><Cell N="X" V="0"/><Cell N="Y" V="0"/></Row>'
        f"</Section><Text><cp IX=\"0\"/>{esc(text)}</Text></Shape>"
    )


def build_page(diagram: Diagram, page_w: float, page_h: float) -> str:
    diagram.layout()
    ids: dict[str, int] = {}
    nid = 1
    shapes: list[str] = []
    connects: list[str] = []

    # header band
    header = Node("__hdr", diagram.title, layer=-1, col=0, fill="#1F2937",
                  line="#1F2937", text_color="#FFFFFF", width=page_w - 1.0, height=0.62)
    header.x = 0.5
    header.y = 0.35
    header.w = page_w - 1.0
    header.h = 0.62
    ids["__hdr"] = nid
    shapes.append(_shape_xml(header, page_h, nid))
    nid += 1

    if diagram.subtitle:
        sub = Node("__sub", diagram.subtitle, layer=-1, col=0, fill="#F1F5F9",
                   line="#E2E8F0", text_color="#334155", width=page_w - 1.0, height=0.40)
        sub.x = 0.5
        sub.y = 1.03
        sub.w = page_w - 1.0
        sub.h = 0.40
        ids["__sub"] = nid
        shapes.append(_shape_xml(sub, page_h, nid))
        nid += 1

    for nd in diagram.nodes.values():
        ids[nd.nid] = nid
        shapes.append(_shape_xml(nd, page_h, nid))
        nid += 1

    for edge in diagram.edges:
        s = diagram.nodes[edge.src]
        d = diagram.nodes[edge.dst]
        label = edge.label
        if not label:
            tail = s.label.split("\n")[-1].strip()
            if tail.startswith("[") and tail.endswith("]"):
                label = tail

        if edge.route in ("skip", "elbow_l"):
            if edge.route == "skip":
                mid_x = (s.x + s.w + d.x) / 2.0
            else:
                mid_x = d.x
            sy = s.y + s.h * 0.5
            dy = d.y + d.h * 0.5
            if edge.route == "elbow_l":
                pts = [(s.x + s.w * 0.5, s.y + s.h), (s.x + s.w * 0.5, dy), (d.x, dy)]
                ly = min(s.y + s.h, dy) + 0.06
                lx = s.x + s.w * 0.5 + 0.08
            else:
                pts = [(s.x + s.w, sy), (mid_x, sy), (mid_x, dy), (d.x, dy)]
                ly = (sy + dy) / 2.0 - 0.11
                lx = mid_x + 0.07
            shapes.append(_polyline_xml(nid, pts, page_h, edge))
            nid += 1
            if label:
                shapes.append(_label_xml(nid, label, lx, ly, page_h))
                nid += 1
            continue

        vertical = edge.route == "auto" and d.layer != s.layer
        if edge.route == "v":
            vertical = True
        elif edge.route == "h":
            vertical = False
        if vertical:
            if d.y >= s.y + s.h:
                x1, y1 = s.x + s.w * 0.5, s.y + s.h
                x2, y2 = d.x + d.w * 0.5, d.y
            else:
                x1, y1 = s.x + s.w * 0.5, s.y
                x2, y2 = d.x + d.w * 0.5, d.y + d.h
        else:
            if d.x >= s.x + s.w:
                x1, y1 = s.x + s.w, s.y + s.h * 0.5
                x2, y2 = d.x, d.y + d.h * 0.5
            else:
                x1, y1 = s.x, s.y + s.h * 0.5
                x2, y2 = d.x + d.w, d.y + d.h * 0.5

        cid = nid
        shapes.append(_connector_xml(cid, x1, y1, x2, y2, page_h, edge,
                                     ids[edge.src], ids[edge.dst], label))
        connects.append(
            f'<Connect FromSheet="{cid}" FromCell="EndX" FromPart="12" '
            f'ToSheet="{ids[edge.dst]}" ToCell="PinX" ToPart="3"/>'
        )
        connects.append(
            f'<Connect FromSheet="{cid}" FromCell="BeginX" FromPart="9" '
            f'ToSheet="{ids[edge.src]}" ToCell="PinX" ToPart="3"/>'
        )
        nid += 1

    body = (
        "<?xml version='1.0' encoding='UTF-8'?>\n"
        f'<PageContents xmlns="{NS}" xml:space="preserve"><Shapes>'
        + "".join(shapes)
        + "</Shapes><Connects>"
        + "".join(connects)
        + "</Connects></PageContents>"
    )
    return body


def build_pages(page_name: str, page_w: float, page_h: float) -> str:
    return (
        "<?xml version='1.0' encoding='UTF-8'?>\n"
        f'<Pages xmlns="{NS}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xml:space="preserve">'
        f'<Page ID="0" NameU="{esc(page_name)}" Name="{esc(page_name)}" ViewScale="-1" '
        f'ViewCenterX="{page_w/2:.4f}" ViewCenterY="{page_h/2:.4f}">'
        '<PageSheet LineStyle="0" FillStyle="0" TextStyle="0">'
        f'<Cell N="PageWidth" V="{page_w:.4f}"/>'
        f'<Cell N="PageHeight" V="{page_h:.4f}"/>'
        '<Cell N="ShdwOffsetX" V="0.1181102362204724"/>'
        '<Cell N="ShdwOffsetY" V="-0.1181102362204724"/>'
        '<Cell N="PageScale" V="0.03937007874015748" U="MM"/>'
        '<Cell N="DrawingScale" V="0.03937007874015748" U="MM"/>'
        '<Cell N="DrawingSizeType" V="3"/>'
        '<Cell N="DrawingScaleType" V="0"/>'
        '<Cell N="InhibitSnap" V="0"/>'
        '<Cell N="PageLockReplace" V="0" U="BOOL"/>'
        '<Cell N="PageLockDuplicate" V="0" U="BOOL"/>'
        '<Cell N="UIVisibility" V="0"/>'
        '<Cell N="ShdwType" V="0"/>'
        '<Cell N="ShdwObliqueAngle" V="0"/>'
        '<Cell N="ShdwScaleFactor" V="1"/>'
        '<Cell N="DrawingResizeType" V="0"/>'
        '<Cell N="PageShapeSplit" V="1"/>'
        '<Section N="Layer"><Row IX="0">'
        '<Cell N="Name" V="Connector"/><Cell N="Color" V="255"/><Cell N="Status" V="0"/>'
        '<Cell N="Visible" V="1"/><Cell N="Print" V="1"/><Cell N="Active" V="0"/>'
        '<Cell N="Lock" V="0"/><Cell N="Snap" V="1"/><Cell N="Glue" V="1"/>'
        '<Cell N="NameUniv" V="Connector"/><Cell N="ColorTrans" V="0"/>'
        '</Row></Section></PageSheet><Rel r:id="rId1"/></Page></Pages>'
    )


def build_app(page_name: str) -> str:
    return (
        "<?xml version='1.0' encoding='UTF-8'?>\n"
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        "<Template/><Application>Microsoft Visio</Application><ScaleCrop>false</ScaleCrop>"
        '<HeadingPairs><vt:vector size="4" baseType="variant">'
        "<vt:variant><vt:lpstr>Pages</vt:lpstr></vt:variant><vt:variant><vt:i4>1</vt:i4></vt:variant>"
        "<vt:variant><vt:lpstr>Masters</vt:lpstr></vt:variant><vt:variant><vt:i4>1</vt:i4></vt:variant>"
        "</vt:vector></HeadingPairs>"
        f'<TitlesOfParts><vt:vector size="2" baseType="lpstr"><vt:lpstr>{esc(page_name)}</vt:lpstr>'
        "<vt:lpstr>Dynamic connector</vt:lpstr></vt:vector></TitlesOfParts>"
        "<Manager/><Company/><LinksUpToDate>false</LinksUpToDate><SharedDoc>false</SharedDoc>"
        "<HyperlinkBase/><HyperlinksChanged>false</HyperlinksChanged><AppVersion>16.0000</AppVersion></Properties>"
    )


# skeleton parts reused verbatim from a reference file
KEEP_PARTS = [
    "[Content_Types].xml",
    "_rels/.rels",
    "docProps/core.xml",
    "visio/_rels/document.xml.rels",
    "visio/document.xml",
    "visio/masters/_rels/masters.xml.rels",
    "visio/masters/master1.xml",
    "visio/masters/masters.xml",
    "visio/pages/_rels/page1.xml.rels",
    "visio/pages/_rels/pages.xml.rels",
    "visio/windows.xml",
]


def write_vsdx(diagram: Diagram, out_path: Path, template: Path) -> None:
    page_w, page_h = diagram.page_size()
    page_xml = build_page(diagram, page_w, page_h)
    pages_xml = build_pages(diagram.page_title, page_w, page_h)
    app_xml = build_app(diagram.page_title)

    with zipfile.ZipFile(template) as z:
        kept = {name: z.read(name) for name in KEEP_PARTS}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in kept.items():
            if name == "docProps/app.xml":
                continue
            z.writestr(name, data)
        z.writestr("visio/pages/page1.xml", page_xml.encode("utf-8"))
        z.writestr("visio/pages/pages.xml", pages_xml.encode("utf-8"))
        z.writestr("docProps/app.xml", app_xml.encode("utf-8"))
