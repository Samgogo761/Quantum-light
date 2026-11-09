from __future__ import annotations

import shutil
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt


ROOT = Path(r"C:\Users\26507\Documents\New_SBEs\Quantum-light")
PPT = ROOT / "docs" / "stage_report_literature_enhanced_20260625_修改版_20260702.pptx"
BACKUP = ROOT / "docs" / "stage_report_literature_enhanced_20260625_backup_20260702.pptx"


def rgb(hex_color: str) -> RGBColor:
    h = hex_color.lstrip("#")
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


COLORS = {
    "ink": "1F2937",
    "muted": "5B6475",
    "blue": "2357A5",
    "green": "2F7D54",
    "purple": "6D54A3",
    "red": "B33B3B",
    "amber": "A56A1E",
    "line": "C7D0DD",
    "bg": "F7F8FB",
}


def replace_text(slide, replacements: dict[str, str]) -> None:
    for shape in slide.shapes:
        if not hasattr(shape, "text") or not shape.text:
            continue
        text = shape.text
        new = text
        for old, repl in replacements.items():
            new = new.replace(old, repl)
        if new != text:
            shape.text = new


def add_text(slide, text: str, x, y, w, h, size=16, bold=False, color="ink") -> None:
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = "Microsoft YaHei"
    run.font.color.rgb = rgb(COLORS[color])


def add_card(slide, x, y, w, h, fill="FFFFFF", line="D7DEEB") -> None:
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = rgb(fill)
    shp.line.color.rgb = rgb(line)
    shp.line.width = Pt(0.8)


def add_note(slide, title: str, body: str, x=0.65, y=6.35, w=12.0, h=0.72, fill="FFF7ED", accent="A56A1E") -> None:
    add_card(slide, x, y, w, h, fill=fill, line="E4C99F")
    add_text(slide, title, x + 0.16, y + 0.11, 2.1, 0.24, size=10, bold=True, color="amber")
    add_text(slide, body, x + 2.0, y + 0.10, w - 2.2, 0.42, size=10, color="ink")


def add_badges(slide, labels: list[str]) -> None:
    x0 = 9.25
    y = 0.28
    for i, label in enumerate(labels):
        x = x0 + i * 1.18
        shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(1.05), Inches(0.30))
        shp.fill.solid()
        shp.fill.fore_color.rgb = rgb(["2357A5", "6D54A3", "2F7D54"][i % 3])
        shp.line.color.rgb = rgb(["2357A5", "6D54A3", "2F7D54"][i % 3])
        tf = shp.text_frame
        tf.clear()
        p = tf.paragraphs[0]
        run = p.add_run()
        run.text = label
        run.font.size = Pt(9)
        run.font.bold = True
        run.font.name = "Microsoft YaHei"
        run.font.color.rgb = rgb("FFFFFF")


def blank_slide(prs):
    # Some older/converted decks only carry a small custom layout set.
    # Reuse the first available layout and clear its placeholder shapes.
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    for shape in list(slide.shapes):
        slide.shapes._spTree.remove(shape._element)
    return slide


def add_world_map_slide(prs):
    slide = blank_slide(prs)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = rgb(COLORS["bg"])
    add_text(slide, "世界地图定位：成熟脚手架 + 一个别人没接进去的磁性维度", 0.55, 0.42, 11.7, 0.55, 25, True, "ink")
    add_text(slide, "这页回答“相对世界已有工作，增量到底在哪里”。", 0.62, 1.05, 10.8, 0.3, 13, False, "muted")

    rows = [
        ("经典 CrI3 HHG", "Liu 2022：经典激光、速度规范、近共振构型指纹", "lg_cov + T2 + 深亚带隙 + 可量化 N 选择定则"),
        ("量子光 HHG 框架", "Gothelf-Lange-Madsen / Stammer / Ilin：量子光与电流涨落框架", "把磁群非互易维度接入：N-宇称 Coo 通道"),
        ("量子优势边界", "González-Monge / Feist：平均谱可由经典强度分布复现", "R1 诚实停在平均谱；R2/R3 用非互易关联越过边界"),
        ("非互易磁光", "Sun 2019；Xiao 2023；Gao 2023：SHG、磁群、量子几何", "从平均响应推广到双时量子关联与磁序传感"),
    ]
    x0, y0 = 0.62, 1.55
    widths = [2.35, 4.45, 5.25]
    headers = ["维度", "现有工作", "本工作增量"]
    for j, h in enumerate(headers):
        add_card(slide, x0 + sum(widths[:j]), y0, widths[j], 0.52, fill="EAF0FB", line="C7D0DD")
        add_text(slide, h, x0 + sum(widths[:j]) + 0.12, y0 + 0.14, widths[j] - 0.24, 0.18, 11, True, "blue")
    for i, row in enumerate(rows):
        y = y0 + 0.52 + i * 0.88
        for j, cell in enumerate(row):
            add_card(slide, x0 + sum(widths[:j]), y, widths[j], 0.88, fill="FFFFFF", line="D7DEEB")
            add_text(slide, cell, x0 + sum(widths[:j]) + 0.12, y + 0.13, widths[j] - 0.25, 0.58, 10.2, j == 0, "ink" if j != 2 else "green")

    add_card(slide, 0.62, 6.10, 11.95, 0.72, fill="F0F7F5", line="BBD8CB")
    add_text(slide, "一句话", 0.82, 6.28, 1.0, 0.2, 10, True, "green")
    add_text(slide, "R1 是干净的量子光平均谱；R2/R3 的新意在于把磁序 N 写进输出光的二阶量子关联，并把“量子优势”变成可检验的矩问题。", 1.65, 6.24, 10.55, 0.28, 12, True, "ink")
    add_badges(slide, ["创新①", "创新②", "创新③"])
    return slide


def add_final_innovation_slide(prs):
    slide = blank_slide(prs)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = rgb("111827")
    add_text(slide, "收尾：三句话和三项创新", 0.62, 0.65, 10.5, 0.6, 30, True, "blue")
    # White title color workaround: direct text helper lacks custom white key.
    for shape in slide.shapes:
        if hasattr(shape, "text") and shape.text.startswith("收尾"):
            for p in shape.text_frame.paragraphs:
                for r in p.runs:
                    r.font.color.rgb = rgb("FFFFFF")

    statements = [
        "经典验证闭环：lg_cov、带窗、k 网格、P 共轭选择定则与外部 benchmark 已经支撑 R1 计算。",
        "R1 诚实定位：BSV 平均谱增强是量子光 HHG 的第一层结果，但可被经典强度分布复现。",
        "R2/R3 才是突破口：Cconn、Σ、Coo 与 QFI 把问题推进到输出光量子统计和磁序传感。",
    ]
    for i, s in enumerate(statements):
        y = 1.55 + i * 0.78
        add_card(slide, 0.8, y, 11.6, 0.58, fill="172235", line="33425D")
        add_text(slide, f"{i+1}. {s}", 1.05, y + 0.14, 11.0, 0.26, 14.5, False, "ink")
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.startswith(f"{i+1}."):
                for p in shape.text_frame.paragraphs:
                    for r in p.runs:
                        r.font.color.rgb = rgb("FFFFFF")

    innovations = [
        ("创新①", "强度谱 -> 输出光量子统计谱"),
        ("创新②", "磁序 N 调控 Bogoliubov 配对谱"),
        ("创新③", "量子优势 -> 经典矩问题无解"),
    ]
    for i, (k, v) in enumerate(innovations):
        x = 0.9 + i * 4.05
        add_card(slide, x, 4.30, 3.5, 1.35, fill=["EAF0FB", "F1ECFB", "EFF8F1"][i], line="6D54A3")
        add_text(slide, k, x + 0.22, 4.55, 1.0, 0.25, 16, True, ["blue", "purple", "green"][i])
        add_text(slide, v, x + 0.22, 4.95, 2.85, 0.36, 15, True, "ink")
    add_text(slide, "执行策略：第一篇先把 R1 写干净；并行用 4-band 最小模型跑通 R2；112-band 第一性原理与 R3 判据作为下一阶段。", 0.85, 6.35, 11.3, 0.32, 14, False, "muted")
    for shape in slide.shapes:
        if hasattr(shape, "text") and shape.text.startswith("执行策略"):
            for p in shape.text_frame.paragraphs:
                for r in p.runs:
                    r.font.color.rgb = rgb("D8E2F2")
    return slide


def reorder_and_drop(prs, desired_original_indices: list[int], extra_indices: list[int]):
    sld_id_lst = prs.slides._sldIdLst
    sld_ids = list(sld_id_lst)
    # desired_original_indices are 1-based positions before adding extra slides.
    desired = [sld_ids[i - 1] for i in desired_original_indices] + [sld_ids[i - 1] for i in extra_indices]
    for sld_id in sld_ids:
        sld_id_lst.remove(sld_id)
    for sld_id in desired:
        sld_id_lst.append(sld_id)


def main():
    if not BACKUP.exists():
        shutil.copy2(PPT, BACKUP)

    prs = Presentation(str(BACKUP))

    # Slide-level text updates by original numbering.
    updates = {
        11: {
            "PT 对称压制占据流形 Berry 曲率 → 反常横向电流通道关闭；偶次 HHG 由带间极化电流携带":
                "PT 压制占据流形 Berry 曲率（Ω_occ 被强压制）→ 反常横向电流通道关闭；偶次由带间极化电流携带",
        },
        12: {
            "量子光 BSV 是什么：相空间、光子配对与强度统计":
                "亮压缩真空 (BSV)：成对光子 + 指数强度分布",
            "亮压缩真空：相空间椭圆、光子只成对、强度服从指数（热式）分布":
                "成对光子、自带强度涨落；高强度尾会更频繁采到强场事件",
        },
        13: {
            "BSV 平均谱 = 对指数强度分布的一维积分":
                "关键简化：BSV 平均谱 = 经典谱对指数强度分布的加权",
            "不必模拟量子场——只需在不同强度上跑经典谱再加权":
                "位移变换后只需做强度加权；但这层仍只是平均谱，不是严格量子优势",
        },
        14: {
            "第一次尝试：蒙特卡洛（MC）PILOT":
                "MC pilot：51 个随机强度能看到增强，但采样噪声不可生产",
            "随机采样 51 个强度——定性看到增强，但统计误差大到不可生产":
                "MC 系统性偏高约 1.5σ；真正瓶颈是采样策略，不是单条轨迹",
        },
        15: {
            "升级：确定性强度求积":
                "确定性强度求积：16 个节点复现指数矩，零采样噪声",
            "在 16 个选定强度各跑一次再加权——零噪声、便宜 >100×":
                "Gauss-Laguerre 型节点把随机噪声变成可控积分误差，成本 >100× 优势",
            "复合 GAUSS–LEGENDRE 求积 + 矩判据":
                "Gauss-Laguerre 型 / 复合确定性强度求积 + 指数矩判据",
        },
        17: {
            "增强的理论框架：为什么是 P!，又为什么压在 P! 之下":
                "解析标尺：微扰阶谐波的 BSV 增强上限严格为 p!",
            "微扰阶谐波的增强上限严格等于 p!；偏离它标记微扰→平台过渡":
                "若 S_n(I) ∝ I^p，则 ⟨S_n⟩/S_n(Ī)=∫e^{-x}x^pdx=p!；偏离即标记微扰→平台过渡",
        },
        20: {
            "换问题：谐波光本身是什么量子态":
                "R2 第一步：位移变换 D(α) 剥离经典激光",
            "位移变换把\"经典激光\"严格剥离，只留量子涨落作微扰":
                "站在量子光 HHG 的通用脚手架上，只把量子涨落作为微扰源",
        },
        21: {
            "源算符：光子可观测量 = 物质算符期望值":
                "R2 第二步：Dyson 展开把谐波光子的源写成电流频率分量",
            "一阶 Dyson 展开把每个谐波光子的\"源\"写成电流的频率分量":
                "输出光统计由电流算符关联决定；一阶给平均场，二阶给涨落",
        },
        22: {
            "怎么算双时关联：量子回归定理":
                "QRT 可落地方案：在 t′ 敲一下 ρ，再用同一 SBE 推进",
            "在 t′ 给 ρ\"敲一下\"，用同一个 SBE 传播子再演化到 t":
                "只新增非厄米辅助传播通道，不重写求解器；非 Markov 情况保留 Keldysh 后备路线",
        },
        23: {
            "能算出的量、CRI₃ 独有的磁性指纹、风险":
                "收获：协方差 Σ 给出量子统计；Coo 给出 N-奇磁性通道",
            "协方差给出压缩/纠缠/光子统计；N-宇称分解给出独有的 C_oo 通道":
                "Σ → 压缩/纠缠/g²/QFI；Coo 是只在磁有序态打开的二阶磁性指纹",
        },
    }
    for idx, reps in updates.items():
        replace_text(prs.slides[idx - 1], reps)

    # Add note boxes to the updated slides.
    add_note(prs.slides[10], "诚实边界", "P 破缺解禁偶次，PT 约束几何；j_anom(t) 尚未单独剥离，不能说已完全排除 Berry 通道。")
    add_note(prs.slides[11], "强度口径", "代码中 BSV 采样的真实平均强度满足 <I>=2·Ī；与经典光比较时必须使用同一平均强度。")
    add_note(prs.slides[12], "世界坐标", "这正对应 González-Monge/Feist 的经典平均边界：R1 做干净，但不声称严格量子优势。")
    add_note(prs.slides[16], "解析标尺", "p! 是指数强度分布的矩，不需拟合；数据低于 p! 表示体系已离开纯微扰幂律。")

    # Innovation badges.
    add_badges(prs.slides[19], ["创新①"])
    add_badges(prs.slides[20], ["创新①"])
    add_badges(prs.slides[21], ["创新①", "创新②"])
    add_badges(prs.slides[22], ["创新②", "创新③"])

    # Add two new slides at the end, then reorder the deck.
    add_world_map_slide(prs)
    add_final_innovation_slide(prs)

    # Keep the external benchmark compact: move original slides 25 and 32 after slide 11, drop 26-31.
    # Add world map after R1/R2/R3 positioning, final innovation slide at the end.
    desired = list(range(1, 12)) + [25, 32] + list(range(12, 20)) + [33] + list(range(20, 25)) + [34]
    reorder_and_drop(prs, desired_original_indices=desired, extra_indices=[])

    prs.save(str(PPT))


if __name__ == "__main__":
    main()


