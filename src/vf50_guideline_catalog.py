"""Thư viện Luật & Case chuẩn Guideline VF-50 (Mục 1-7, Bảng 3 & Bảng 3.1).

Định nghĩa mã luật (rule_code), mô tả điều kiện định lượng, căn cứ Guideline
và các case mẫu thực tế để vừa gắn giải thích cho từng điểm (PointAnnotation),
vừa cung cấp dữ liệu cho UI Modal "Thư viện Luật & Case VF-50".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class GuidelineRuleSpec:
    code: str
    category: str
    title: str
    guideline_ref: str
    default_state: str
    condition_summary: str
    case_example: str


GUIDELINE_RULES: List[GuidelineRuleSpec] = [
    GuidelineRuleSpec(
        code="R_PT_VISIBLE",
        category="point_state",
        title="Điểm nhìn thấy rõ (Visible)",
        guideline_ref="Mục 3 - Bảng 3",
        default_state="visible",
        condition_summary=(
            "Điểm nằm trong khung ảnh, vector pháp tuyến 3D hướng về phía camera, "
            "không bị gọng/kính đen/tóc/khẩu trang/sống mũi che khuất."
        ),
        case_example="Mặt thẳng hoặc nghiêng vừa (|Yaw| < 35°), không bị vật cản che lên điểm.",
    ),
    GuidelineRuleSpec(
        code="R_PT_CLOSED_EYE_VISIBLE",
        category="glasses",
        title="Mắt nhắm hoặc nheo vẫn Visible",
        guideline_ref="Mục 4.1 (Trường hợp đặc biệt - Mắt nhắm)",
        default_state="visible",
        condition_summary=(
            "Khi tài xế nhắm mắt hoặc chớp mắt, mi trên và mi dưới khép lại nhưng toàn bộ "
            "8 điểm viền mắt vẫn nhìn thấy rõ trên khe mí -> giữ Visible (occluded=0, outside=0)."
        ),
        case_example="Ảnh nhắm mắt ngủ gật / chớp mắt trong cabin IR không đeo kính râm.",
    ),
    GuidelineRuleSpec(
        code="R_PT_CLOSED_MOUTH_VISIBLE",
        category="point_state",
        title="Miệng ngậm vẫn giữ moitrong Visible",
        guideline_ref="Mục 4.2 (Trường hợp đặc biệt - Miệng ngậm)",
        default_state="visible",
        condition_summary=(
            "Khi miệng ngậm khép chặt, cả 12 điểm moingoai và 8 điểm moitrong đều giữ Visible; "
            "cụm moitrong bám dọc theo đường khép môi, tuyệt đối không gắn Hidden/Occluded/Outside."
        ),
        case_example="Tài xế ngậm miệng bình thường (moingoai & moitrong Visible đầy đủ).",
    ),
    GuidelineRuleSpec(
        code="R_PT_GLASSES_CLEAR",
        category="glasses",
        title="Kính trong suốt / kính cận nhìn thấy mắt",
        guideline_ref="Mục 4.3a (Đeo kính trong)",
        default_state="visible",
        condition_summary=(
            "BiSeNet phát hiện tròng kính nhưng độ tương phản mắt bình thường (ctr >= 9.5) "
            "hoặc nhìn rõ vùng da/mống mắt xuyên qua tròng (eye_vis >= 0.022) -> Mắt giữ Visible."
        ),
        case_example="Đeo kính cận không gọng hoặc gọng mảnh nhìn xuyên qua tròng thấy đồng tử.",
    ),
    GuidelineRuleSpec(
        code="R_PT_GLASSES_FRAME_RIM",
        category="glasses",
        title="Gọng kính / viền kính đè trực tiếp lên điểm",
        guideline_ref="Mục 4.3c (Gọng kính che cục bộ)",
        default_state="occluded",
        condition_summary=(
            "Điểm (thường là lông mày hoặc góc mắt) nằm dưới phần gọng/viền kính tối, "
            "vẫn nội suy/ước lượng được chính xác vị trí trên cấu trúc mặt -> Occluded=1."
        ),
        case_example="Viền trên kính râm hoặc gọng kính dày chạy ngang hàng lông mày longmaytrai/longmayphai.",
    ),
    GuidelineRuleSpec(
        code="R_PT_GLASSES_GLARE_OCC",
        category="glasses",
        title="Phản quang / lóa IR trên tròng kính che mờ cục bộ",
        guideline_ref="Mục 4.3d (Lóa kính cục bộ)",
        default_state="occluded",
        condition_summary=(
            "Ánh sáng hoặc đèn IR phản xạ làm mờ một số điểm mắt trên tròng kính trong, "
            "nhưng vẫn còn định vị được tổng thể bộ mắt -> các điểm bị lóa gắn Occluded=1."
        ),
        case_example="Kính trắng bị vệt lóa đèn đường hoặc LED IR phản chiếu lên nửa tròng kính.",
    ),
    GuidelineRuleSpec(
        code="R_PT_SUNGLASSES_BLACK_OUT",
        category="glasses",
        title="Kính râm đen / kính tối màu che mất hoàn toàn mắt",
        guideline_ref="Mục 4.3b (Đeo kính râm tối)",
        default_state="outside",
        condition_summary=(
            "Kính râm đen che kín cả vùng mắt (eye_vis < 0.022, độ tương phản tròng đồng nhất ctr < 9.5 "
            "hoặc cặp kính râm hai bên tối đồng thời), không thể quan sát/ước lượng vị trí mí mắt -> "
            "toàn bộ mattrai (14..21) & matphai (22..29) gắn Outside=1 (occluded=0)."
        ),
        case_example="Tài xế đeo kính râm đen (media sunglasses): 16 điểm mắt Outside, lông mày dưới viền gọng Occluded, mũi & miệng Visible.",
    ),
    GuidelineRuleSpec(
        code="R_PT_SELF_OCC_3D_NOSE",
        category="self_occlusion",
        title="Tự che khuất 3D do sống mũi / tháp mũi cản tia nhìn",
        guideline_ref="Mục 3 - Định nghĩa Bị che (Tự che bởi mũi)",
        default_state="occluded",
        condition_summary=(
            "Truy vết tia 3D (Raycasting) từ camera tới vertex 3D của điểm đi xuyên qua "
            "bề mặt tháp mũi (nasal pyramid) trước khi tới mặt phẳng khuôn mặt ở phía xa camera."
        ),
        case_example="Nghiêng đầu ~38°-50°: khóe mắt trong hoặc đầu lông mày phía xa bị sống mũi che khuất.",
    ),
    GuidelineRuleSpec(
        code="R_PT_SELF_OCC_3D_HORIZON",
        category="self_occlusion",
        title="Tự che khuất 3D do mặt cong quay ra rìa (Horizon)",
        guideline_ref="Mục 3 - Định nghĩa Bị che (Góc nghiêng 3D)",
        default_state="occluded",
        condition_summary=(
            "Tích vô hướng vector pháp tuyến 3D tại điểm với vector hướng về camera âm (cos < 0) "
            "khi xoay đầu nghiêng vừa/sâu nhưng chưa đạt ngưỡng mất hoàn toàn bộ phận."
        ),
        case_example="Nghiêng đầu ~40°-50° không vật cản: nửa mắt/lông mày phía xa camera bị che một phần -> Occluded=1.",
    ),
    GuidelineRuleSpec(
        code="R_PT_SELF_OUT_EXTREME_PROFILE",
        category="self_occlusion",
        title="Góc nghiêng cực đại (Extreme Profile) khuất hẳn nửa mặt",
        guideline_ref="Mục 3 & Bảng 3.1 (Ngoài vùng nhìn thấy)",
        default_state="outside",
        condition_summary=(
            "Mặt nghiêng gần vuông góc (|Yaw| cực đại hoặc bề mặt 3D khuất sâu sau sống mũi): "
            "lông mày và mắt phía xa camera biến mất hoàn toàn, không thể suy đoán -> Outside=1."
        ),
        case_example="Tài xế ngoái nhìn sang cửa sổ (|Yaw| >= 58°): longmay + mat phía xa camera thành 13 điểm Outside.",
    ),
    GuidelineRuleSpec(
        code="R_PT_MASK_OCC",
        category="parsing_occlusion",
        title="Khẩu trang / vật che nửa dưới mặt (ước lượng được)",
        guideline_ref="Mục 4.4a (Đeo khẩu trang / vật cản)",
        default_state="occluded",
        condition_summary=(
            "BiSeNet phát hiện khẩu trang/vải che lên mũi hoặc miệng nhưng vẫn đủ bối cảnh định vị "
            "một phần điểm -> đánh Occluded=1 cho các điểm bị che cục bộ."
        ),
        case_example="Khẩu trang kéo trễ dưới mũi hoặc tay chạm một góc miệng.",
    ),
    GuidelineRuleSpec(
        code="R_PT_MASK_OUT",
        category="parsing_occlusion",
        title="Khẩu trang / vật che che kín hoàn toàn bộ phận",
        guideline_ref="Mục 4.4b & Bảng 3.1 (Mất dấu hoàn toàn)",
        default_state="outside",
        condition_summary=(
            "Khẩu trang y tế / khăn che kín toàn bộ mũi dưới hoặc toàn bộ miệng khiến số điểm Visible "
            "nhỏ hơn ngưỡng Bảng 3.1 -> toàn bộ skeleton chuyển Outside=1."
        ),
        case_example="Đeo khẩu trang kín mũi miệng: moingoai (12đ), moitrong (8đ) và các điểm mũi bị che gắn Outside=1.",
    ),
    GuidelineRuleSpec(
        code="R_PT_HAIR_OCC",
        category="parsing_occlusion",
        title="Tóc / phụ kiện che cục bộ lên lông mày hoặc mắt",
        guideline_ref="Mục 3 - Bị che bởi tóc/mũ",
        default_state="occluded",
        condition_summary=(
            "BiSeNet phát hiện tóc/vải phủ lên một vài điểm lông mày hoặc viền mắt nhưng phần còn lại "
            "của bộ phận vẫn thấy rõ đủ giữ ngưỡng Bảng 3.1 -> điểm bị phủ gắn Occluded=1."
        ),
        case_example="Tóc mái lòa xòa che 1-2 điểm đầu hoặc đuôi lông mày.",
    ),
    GuidelineRuleSpec(
        code="R_PT_HAIR_OUT",
        category="parsing_occlusion",
        title="Tóc / tay che kín toàn bộ bộ phận mắt hoặc lông mày",
        guideline_ref="Mục 3 & Bảng 3.1",
        default_state="outside",
        condition_summary=(
            "Tóc dày, mũ hoặc bàn tay che kín quá ngưỡng Bảng 3.1 khiến không thể xác định được bộ phận -> Outside=1."
        ),
        case_example="Lấy tay che mắt hoặc tóc phủ kín hoàn toàn một bên lông mày & mắt.",
    ),
    GuidelineRuleSpec(
        code="R_PT_OUT_OF_FRAME",
        category="out_of_frame",
        title="Điểm vượt ra ngoài biên ảnh (Out of Frame / Crop)",
        guideline_ref="Mục 3 - Ngoài khung hình",
        default_state="outside",
        condition_summary=(
            "Tọa độ (x, y) nằm ngoài giới hạn ảnh [0, width) x [0, height) -> bắt buộc Outside=1, Occluded=0."
        ),
        case_example="Đầu người lái bị cắt sát mép trái/phải/trên của camera cabin.",
    ),
    GuidelineRuleSpec(
        code="R_SKEL_TABLE31_COLLAPSE",
        category="skeleton_rule",
        title="Sập cả bộ phận theo ngưỡng tối thiểu Bảng 3.1",
        guideline_ref="Mục 3 - Bảng 3.1 (Quy tắc mức Skeleton)",
        default_state="outside",
        condition_summary=(
            "Mỗi skeleton yêu cầu số điểm Visible >= ngưỡng tối thiểu (longmay 3/5, songmui 2/4, mat 4/8, "
            "moingoai 6/12, moitrong 4/8). Nếu không đạt ngưỡng -> toàn bộ các điểm của skeleton đó phải chuyển Outside=1."
        ),
        case_example="Nếu 5/8 điểm mắt bị che/ngoài khung -> chỉ còn 3/8 Visible (< 4) => toàn bộ 8 điểm mắt chuyển Outside=1.",
    ),
    GuidelineRuleSpec(
        code="R_SKEL_COUPLED_LIPS",
        category="skeleton_rule",
        title="Đồng bộ ràng buộc moingoai & moitrong",
        guideline_ref="Mục 3 - Bảng 3.1 (Môi ngoài & Môi trong)",
        default_state="outside",
        condition_summary=(
            "moingoai (30..41) và moitrong (42..49) luôn đi cặp: nếu một trong hai sập thành toàn bộ Outside "
            "thì skeleton còn lại cũng đồng bộ thành Outside=1 (hoặc cùng tồn tại)."
        ),
        case_example="Khẩu trang che hoàn toàn moingoai -> cả 12 điểm moingoai + 8 điểm moitrong đều thành Outside=1.",
    ),
    GuidelineRuleSpec(
        code="R_MANUAL_EDIT",
        category="manual",
        title="Chỉnh sửa thủ công bởi người gán nhãn",
        guideline_ref="Thao tác QA / Fine-tune thủ công",
        default_state="mixed",
        condition_summary=(
            "Trạng thái Occluded hoặc Outside được gán nhãn viên điều chỉnh trực tiếp bằng chuột phải / phím tắt."
        ),
        case_example="Người dùng chuột phải hoặc bấm O / X để chuyển trạng thái cho từng điểm trên canvas.",
    ),
]

RULE_BY_CODE: Dict[str, GuidelineRuleSpec] = {r.code: r for r in GUIDELINE_RULES}

TABLE31_RULES: List[Dict[str, Any]] = [
    {"skeleton": "longmaytrai", "label": "Lông mày trái", "ids": "0..4", "total": 5, "min_visible": 3},
    {"skeleton": "longmayphai", "label": "Lông mày phải", "ids": "5..9", "total": 5, "min_visible": 3},
    {"skeleton": "songmui", "label": "Sống mũi", "ids": "10..13", "total": 4, "min_visible": 2},
    {"skeleton": "mattrai", "label": "Mắt trái", "ids": "14..21", "total": 8, "min_visible": 4},
    {"skeleton": "matphai", "label": "Mắt phải", "ids": "22..29", "total": 8, "min_visible": 4},
    {"skeleton": "moingoai", "label": "Môi ngoài", "ids": "30..41", "total": 12, "min_visible": 6},
    {"skeleton": "moitrong", "label": "Môi trong", "ids": "42..49", "total": 8, "min_visible": 4},
]

SPECIAL_CASES: List[Dict[str, Any]] = [
    {
        "title": "Kính râm đen ở góc nghiêng nhẹ/vừa (~28°)",
        "expected": "16 điểm mắt Outside (R_PT_SUNGLASSES_BLACK_OUT), lông mày dưới viền gọng Occluded (R_PT_GLASSES_FRAME_RIM), mũi + miệng 24/24 Visible.",
        "note": "Tuyệt đối không đánh lây Occluded lên khóe miệng 35, 36, 37, 46 khi miệng hoàn toàn lộ rõ.",
    },
    {
        "title": "Kính cận trong suốt / IR nhìn thấy đồng tử",
        "expected": "Mắt giữ Visible (R_PT_GLASSES_CLEAR); chỉ điểm bị gọng dày che hoặc lóa IR nặng mới gắn Occluded.",
        "note": "BiSeNet có thể phân loại vùng kính (eye_g), nhưng độ tương phản mắt vẫn cao (>= 9.5) nên không bị đánh nhầm thành kính râm.",
    },
    {
        "title": "Nghiêng đầu IR cực đại / Extreme Profile (~58° - 90°)",
        "expected": "Lông mày + mắt phía xa camera (13 điểm: 5+8) thành Outside (R_PT_SELF_OUT_EXTREME_PROFILE), mũi/miệng phía xa có thể Occluded.",
        "note": "Suy luận từ Yaw + tỷ lệ dẹt 2D + truy vết tia 3D qua tháp mũi.",
    },
    {
        "title": "Mắt nhắm / Miệng ngậm không vật cản",
        "expected": "Toàn bộ giữ Visible (R_PT_CLOSED_EYE_VISIBLE / R_PT_CLOSED_MOUTH_VISIBLE), không dùng Hidden.",
        "note": "CVAT VF-50 chỉ hỗ trợ 3 trạng thái: Visible (0,0), Occluded (1,0), Outside (0,1).",
    },
    {
        "title": "Đeo khẩu trang che mũi dưới & miệng",
        "expected": "moingoai + moitrong sập hoàn toàn thành Outside (R_PT_MASK_OUT + R_SKEL_COUPLED_LIPS), chóp/cánh mũi bị che Occluded hoặc Outside.",
        "note": "Lông mày & mắt phía trên vẫn giữ Visible bình thường.",
    },
]

QC_FLAG_DESCRIPTIONS: List[Dict[str, str]] = [
    {"flag": "LOW_RES", "meaning": "Ảnh khuôn mặt có kích thước hoặc mật độ pixel thấp dưới ngưỡng khuyến nghị."},
    {"flag": "DARK_LOW_CONTRAST", "meaning": "Ảnh tối / thiếu tương phản (thường gặp trên camera IR ban đêm)."},
    {"flag": "MOTION_BLUR", "meaning": "Ảnh bị nhòe chuyển động do rung lắc xe hoặc quay đầu nhanh."},
    {"flag": "EXTREME_PROFILE_R", "meaning": "Mặt nghiêng mạnh sang phải; nửa mặt phải khuất khỏi camera."},
    {"flag": "EXTREME_PROFILE_L", "meaning": "Mặt nghiêng mạnh sang trái; nửa mặt trái khuất khỏi camera."},
    {"flag": "HEAVY_OCCLUSION_NOSE", "meaning": "Bộ phận songmui bị che/mất dấu nặng (>= 50% điểm)."},
    {"flag": "HEAVY_OCCLUSION_MOUTH", "meaning": "Bộ phận moingoai/moitrong bị che hoặc sập Outside theo Bảng 3.1."},
    {"flag": "SUNGLASSES_OCCLUSION", "meaning": "Phát hiện kính râm tối màu che kín vùng mắt (Mục 4.3b)."},
    {"flag": "NOSE_COLLAPSED_OUTSIDE", "meaning": "Sống mũi không đạt ngưỡng >= 2/4 Visible -> toàn bộ Outside."},
    {"flag": "MOUTH_COLLAPSED_OUTSIDE", "meaning": "Môi không đạt ngưỡng >= 6/12 hoặc >= 4/8 -> đồng bộ toàn bộ Outside."},
]


def get_guideline_catalog_dict() -> Dict[str, Any]:
    """Trả về toàn bộ Thư viện Luật & Case VF-50 dạng JSON-serializable cho Web UI."""
    return {
        "rules": [asdict(r) for r in GUIDELINE_RULES],
        "rule_map": {r.code: asdict(r) for r in GUIDELINE_RULES},
        "table31": TABLE31_RULES,
        "special_cases": SPECIAL_CASES,
        "qc_flags": QC_FLAG_DESCRIPTIONS,
    }
