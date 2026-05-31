#!/usr/bin/env python3
"""Script to improve the academic paper document about solid-state HHG."""

import re
from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from lxml import etree

INPUT_PATH = '/root/.claude/uploads/ed27d24f-ff18-4d97-9d12-16efb3450db5/d0ce837a-__________________________.docx'
OUTPUT_PATH = '/home/user/Quantum-light/综述_修订版.docx'

doc = Document(INPUT_PATH)


def make_paragraph_element(text, style_val='Normal', bold=False):
    """Create a new w:p element with the given text and style."""
    new_para = OxmlElement('w:p')

    # Add paragraph properties with style
    pPr = OxmlElement('w:pPr')
    pStyle = OxmlElement('w:pStyle')
    pStyle.set(qn('w:val'), style_val)
    pPr.append(pStyle)
    new_para.append(pPr)

    # Add run
    r = OxmlElement('w:r')
    if bold:
        rPr = OxmlElement('w:rPr')
        b = OxmlElement('w:b')
        rPr.append(b)
        r.append(rPr)
    t = OxmlElement('w:t')
    t.text = text
    t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    r.append(t)
    new_para.append(r)

    return new_para


def insert_para_before(ref_para_elem, new_para_elem):
    """Insert new_para_elem before ref_para_elem in parent."""
    parent = ref_para_elem.getparent()
    idx = list(parent).index(ref_para_elem)
    parent.insert(idx, new_para_elem)


# ============================================================
# 1. Add Abstract section before "1 引言"
# ============================================================
print("Step 1: Adding abstract section...")

# Find the paragraph with "1 引言"
intro_para = None
for p in doc.paragraphs:
    if p.text.strip() == '1 引言':
        intro_para = p
        break

if intro_para is None:
    print("  WARNING: Could not find '1 引言' paragraph!")
else:
    ref_elem = intro_para._element

    # Abstract content (insert in reverse order since each inserts before ref_elem)
    abstract_items = [
        # (text, style_val, bold)
        ("关键词：固体高次谐波；量子光；二维磁性材料；谷极化；双层反铁磁CrI₃；半导体布洛赫方程；量子电动力学", 'Normal', False),
        ("高次谐波产生（High-order Harmonic Generation, HHG）是强激光场与物质相互作用中最具代表性的非微扰非线性光学过程，也是阿秒科学和极紫外相干光源的重要物理基础。传统固体HHG研究通常将驱动光场视为经典相干态，而近年来量子态光场（如明亮压缩真空，BSV）对强场过程的驱动作用成为全新的研究前沿。本综述以具有丰富磁序结构的双层范德华磁性材料CrI₃为核心平台，系统阐述固体HHG的基本物理机制、二维材料中的对称性选择定则、谷极化的拓扑起源，以及量子光统计特性对谷极化与谐波手性的调制作用。在理论框架方面，本文介绍了第一性原理结合紧束缚模型、半导体布洛赫方程（SBEs）和严格量子电动力学（QED）的多尺度计算方法，并给出了量子光场（BSV）驱动下CrI₃体系中一系列可观测量子效应的理论预言，包括拉比振荡崩塌、截止频率非微扰拓宽和贝里曲率放大量子涨落等。本综述旨在推动阿秒物理、拓扑谷电子学与量子信息科学的深度融合。", 'Normal', False),
        ("摘要", 'Normal', True),
        ("Keywords: solid-state high-harmonic generation; quantum light; two-dimensional magnetic materials; valley polarization; bilayer antiferromagnetic CrI₃; semiconductor Bloch equations; quantum electrodynamics", 'Normal', False),
        ("High-order harmonic generation (HHG) is one of the most representative non-perturbative nonlinear optical processes in intense laser-matter interactions, and serves as an important physical basis for attosecond science and coherent extreme-ultraviolet sources. While conventional solid-state HHG studies typically treat the driving field as a classical coherent state, the role of quantum light fields—such as bright squeezed vacuum (BSV)—in driving strong-field processes has recently emerged as a new research frontier. This review focuses on bilayer van der Waals magnetic material CrI₃ as the central platform, systematically discussing the fundamental physical mechanisms of solid-state HHG, symmetry selection rules in two-dimensional materials, the topological origin of valley polarization, and the modulation of valley polarization and harmonic chirality by quantum photon statistics. On the theoretical side, we present a multi-scale computational framework combining first-principles calculations with tight-binding models, semiconductor Bloch equations (SBEs), and rigorous quantum electrodynamics (QED), and propose theoretical predictions for a series of observable quantum phenomena in CrI₃ driven by BSV, including Rabi oscillation collapse, non-perturbative extension of the cutoff frequency, and Berry curvature amplification of quantum fluctuations. This review aims to promote the deep integration of attosecond physics, topological valley electronics, and quantum information science.", 'Normal', False),
        ("Abstract", 'Normal', True),
    ]

    for text, style_val, bold in abstract_items:
        new_elem = make_paragraph_element(text, style_val, bold)
        insert_para_before(ref_elem, new_elem)

    print("  Abstract section inserted successfully.")


# ============================================================
# 2. Change subsection headings to Heading 2 style
# ============================================================
print("Step 2: Changing subsection headings to Heading 2 style...")

subsection_pattern = re.compile(r'^\d+\.\d+\s')
changed_count = 0

for p in doc.paragraphs:
    text = p.text.strip()
    if subsection_pattern.match(text) and p.style.name == 'Normal':
        # Check it's not "5.4" combined paragraph (will be handled in step 4)
        # Actually we still need to handle the heading part; step 4 will split it
        p.style = doc.styles['Heading 2']
        changed_count += 1

print(f"  Changed {changed_count} subsection headings to Heading 2.")


# ============================================================
# 3. Fix section 4.1 duplicate text
# ============================================================
print("Step 3: Fixing section 4.1 duplicate text...")

fixed = False
for p in doc.paragraphs:
    if '首先，利用密度泛函理论（DFT）[21]（DFT）' in p.text:
        for run in p.runs:
            if '首先，利用密度泛函理论（DFT）[21]（DFT）' in run.text:
                run.text = run.text.replace(
                    '首先，利用密度泛函理论（DFT）[21]（DFT）',
                    '首先，利用密度泛函理论（DFT）[21]'
                )
                fixed = True
                break
        # Also check if the text is split across runs
        if not fixed:
            full_text = p.text
            if '首先，利用密度泛函理论（DFT）[21]（DFT）' in full_text:
                # Rebuild by modifying the XML text nodes directly
                for elem in p._element.iter(qn('w:t')):
                    if elem.text and '首先，利用密度泛函理论（DFT）[21]（DFT）' in elem.text:
                        elem.text = elem.text.replace(
                            '首先，利用密度泛函理论（DFT）[21]（DFT）',
                            '首先，利用密度泛函理论（DFT）[21]'
                        )
                        fixed = True
                        break

if fixed:
    print("  Fixed duplicate (DFT) text in section 4.1.")
else:
    print("  WARNING: Could not find duplicate (DFT) text to fix!")


# ============================================================
# 4. Fix section 5.4 heading - split into heading + content
# ============================================================
print("Step 4: Fixing section 5.4 heading...")

target_54_text = '5.4 核心待检验预言综合上述微观机制'
content_54 = '综合上述微观机制，本课题给出的核心实验可检验预言为：在相同中心波长与平均强度的条件下，仅通过切换驱动光源的统计分布状态（如从泊松分布的相干态切换为超泊松分布的压缩真空态），即可对双层 CrI₃ 中反映 AFM/FM 磁序指纹的奇数/偶数阶高次谐波的偏振对比度产生确定性的调制与频移。'

para_54 = None
for p in doc.paragraphs:
    if p.text.startswith('5.4 核心待检验预言综合上述微观机制'):
        para_54 = p
        break

if para_54 is None:
    print("  WARNING: Could not find 5.4 paragraph to split!")
else:
    # Change existing paragraph to heading with just "5.4 核心待检验预言"
    # Clear all runs and set new text
    for run in para_54.runs:
        run.text = ''

    # Set style to Heading 2
    para_54.style = doc.styles['Heading 2']

    # Set the text via XML
    # Remove existing runs
    for r in para_54._element.findall(qn('w:r')):
        para_54._element.remove(r)

    # Add new run with heading text
    r_elem = OxmlElement('w:r')
    t_elem = OxmlElement('w:t')
    t_elem.text = '5.4 核心待检验预言'
    t_elem.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    r_elem.append(t_elem)
    para_54._element.append(r_elem)

    # Insert content paragraph after the heading
    content_elem = make_paragraph_element(content_54, 'Normal', False)
    # Insert after para_54
    parent = para_54._element.getparent()
    idx = list(parent).index(para_54._element)
    parent.insert(idx + 1, content_elem)

    print("  Section 5.4 heading split successfully.")


# ============================================================
# 5. Fix reference list entries
# ============================================================
print("Step 5: Fixing reference list entries...")

ref_replacements = {
    '[4]': {
        'find_start': '[4] Z. Y. Wei',
        'new_text': '[4] Wei Z Y, Wang Z H, Hao T, Han H N, Chang G Q. Ultrashort pulse laser systems for high-harmonic and attosecond science. 【注：此参考文献信息不完整，请补充完整文献信息，包含期刊名、卷期、页码和DOI】'
    },
    '[5]': {
        'find_start': '[5] P. Agostini',
        'new_text': '[5] Agostini P, DiMauro L F. The physics of attosecond light pulses. Reports on Progress in Physics, 2004, 67(6): 813-855. https://doi.org/10.1088/0034-4885/67/6/R01'
    },
    '[6]': {
        'find_start': '[6] Corkum, P. B., Krausz',
        'new_text': '[6] Corkum P B. Plasma perspective on strong field multiphoton ionization. Physical Review Letters, 1993, 71(13): 1994-1997. https://doi.org/10.1103/PhysRevLett.71.1994'
    },
    '[8]': {
        'find_start': '[8] Agostini, P., et al. (2001)',
        'new_text': '[8] Hentschel M, Kienberger R, Spielmann C, Reider G A, Milosevic N, Brabec T, Corkum P, Heinzmann U, Drescher M, Krausz F. Attosecond metrology. Nature, 2001, 414(6863): 509-513. https://doi.org/10.1038/35107000'
    },
    '[20]': {
        'find_start': '[20] Sun, D., Rao, R.',
        'new_text': '[20] Sun Z, Yi Y, Song T, Clark G, Huang B, Shan Y, Wu S, Huang D, Gao C, Chen Z, McGuire M, Cao T, Xiao D, Liu W T, Yao W, Xu X, Wu S. Giant nonreciprocal second-harmonic generation from antiferromagnetic bilayer CrI₃. Nature, 2019, 572(7770): 497-501. https://doi.org/10.1038/s41586-019-1445-3'
    },
    '[21]': {
        'find_start': '[21] Huang, B., Clark, G.',
        'new_text': '[21] Kohn W, Sham L J. Self-consistent equations including exchange and correlation effects. Physical Review, 1965, 140(4A): A1133-A1138. https://doi.org/10.1103/PhysRev.140.A1133'
    },
    '[22]': {
        'find_start': '[22] Seyler, K. L., Zhong',
        'new_text': '[22] Marzari N, Vanderbilt D. Maximally localized generalized Wannier functions for composite energy bands. Physical Review B, 1997, 56(20): 12847-12865. https://doi.org/10.1103/PhysRevB.56.12847'
    },
}

for ref_key, info in ref_replacements.items():
    found = False
    for p in doc.paragraphs:
        if p.text.startswith(info['find_start']):
            # Clear all runs
            for r in p._element.findall(qn('w:r')):
                p._element.remove(r)
            # Add new run
            r_elem = OxmlElement('w:r')
            t_elem = OxmlElement('w:t')
            t_elem.text = info['new_text']
            t_elem.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
            r_elem.append(t_elem)
            p._element.append(r_elem)
            found = True
            print(f"  Fixed reference {ref_key}.")
            break
    if not found:
        print(f"  WARNING: Could not find reference {ref_key} (looking for: {info['find_start']!r})")


# ============================================================
# 6. Add missing references [23]-[31] after [22]
# ============================================================
print("Step 6: Adding references [23]-[31]...")

new_refs = [
    "[23] Huang B, Clark G, Navarro-Moratalla E, Klein D R, Cheng R, Seyler K L, Zhong D, Schmidgall E, McGuire M A, Cobden D H, Yao W, Xiao D, Jarillo-Herrero P, Xu X. Layer-dependent ferromagnetism in a van der Waals crystal down to the monolayer limit. Nature, 2017, 546(7657): 270-273. https://doi.org/10.1038/nature22391",
    "[24] Seyler K L, Zhong D, Huang B, Linpeng X, Wilson N P, Taniguchi T, Watanabe K, Yao W, Xiao D, McGuire M A, Cobden D H, Xu X. Ligand-field helicity for spin-momentum locking in a two-dimensional ferromagnet. Nature Physics, 2018, 14(3): 277-281. https://doi.org/10.1038/nphys4058",
    "[25] Stammer P, Rivera-Dean J, Lamprou T, Pisanty E, Ciappina M F, Lewenstein M, Tzallas P. High photon number entangled states and coherent state superposition from the extreme ultraviolet to the far infrared. Physical Review Letters, 2022, 128(12): 123603. https://doi.org/10.1103/PhysRevLett.128.123603",
    "[26] Even Tzur M, Birk M, Gorlach A, Krüger M, Kaminer I, Cohen O. Photon-statistics force in ultrafast electron dynamics. Nature Photonics, 2023, 17(6): 501-509. https://doi.org/10.1038/s41566-023-01215-y",
    "[27] Gothelf R V, Lange C S, Madsen L B. High-harmonic generation in a crystal driven by quantum light. arXiv:2502.11803, 2025. https://arxiv.org/abs/2502.11803",
    "[28] 【待补全：图10对应的集成光子芯片参考文献，请填写完整信息】",
    "[29] Song T, Cai X, Tu M W Y, Zhang X, Huang B, Wilson N P, Seyler K L, Zhu L, Taniguchi T, Watanabe K, McGuire M A, Cobden D H, Xiao D, Yao W, Xu X. Giant tunneling magnetoresistance in spin-filter van der Waals heterostructures. Science, 2018, 360(6394): 1214-1218. https://doi.org/10.1126/science.aar4851",
    "[30] 【待补全：图11对应的HHG泵浦探测相干衍射成像参考文献，请填写完整信息】",
    "[31] Klein D R, MacNeill D, Lado J L, Soriano D, Navarro-Moratalla E, Watanabe K, Taniguchi T, Manni S, Canfield P, Fernández-Rossier J, Jarillo-Herrero P. Probing magnetism in 2D van der Waals crystalline insulators via electron tunneling. Science, 2018, 360(6394): 1218-1222. https://doi.org/10.1126/science.aar3617",
]

# Find the [22] reference paragraph (now updated)
ref22_para = None
for p in doc.paragraphs:
    if p.text.startswith('[22] Marzari'):
        ref22_para = p
        break

if ref22_para is None:
    print("  WARNING: Could not find [22] reference to insert after!")
else:
    ref22_elem = ref22_para._element
    parent = ref22_elem.getparent()
    idx = list(parent).index(ref22_elem)

    for i, ref_text in enumerate(new_refs):
        new_elem = make_paragraph_element(ref_text, 'Normal', False)
        parent.insert(idx + 1 + i, new_elem)

    print(f"  Added {len(new_refs)} new references [23]-[31].")


# ============================================================
# Save the document
# ============================================================
print(f"\nSaving to {OUTPUT_PATH}...")
doc.save(OUTPUT_PATH)
print("Done!")

# Verify
doc2 = Document(OUTPUT_PATH)
print(f"\nVerification - total paragraphs: {len(doc2.paragraphs)}")

# Check abstract
for i, p in enumerate(doc2.paragraphs[:20]):
    print(f"  {i}: [{p.style.name}] {repr(p.text[:80])}")
