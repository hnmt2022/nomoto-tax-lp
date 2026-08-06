#!/usr/bin/env python3
"""
野本裕美 訪問相談LP用画像生成スクリプト

Gemini APIを使用し、公認会計士・税理士 野本裕美の訪問相談サービス
（年金生活の税金チェック／暮らしと財産の現在地整理）向けの
LP画像素材を生成する。

画像の基本方針:
  - ブランドカラー（深紺・クリーム、アクセントに柔らかな金）
  - シニアを「支えられる側」として一律に描かず、意思を持つ主体として表現
  - 相続・税金・死・老いを不安をあおる形で扱わない。「争族」を煽る表現は厳禁
  - 財産の多さを前提・誇示する演出（豪邸、宝飾品、札束等）を避ける
  - 画像内の文字は原則入れず、HTML/CSSで重ねる

使用例:
    python lp-image-create.py --preset hero \
        --prompt "明るい自宅のリビングで書類を確認しながら穏やかに話す60代女性と訪問した税理士" \
        --output ../assets/images/hero-bg.png

    python lp-image-create.py --preset consultation \
        --prompt "テーブルに年金や税金の書類を置き、訪問した税理士と落ち着いて話す女性" \
        --output ../assets/images/tax-check-consultation.png

    python lp-image-create.py --preset doc-amount \
        --prompt "A4ファイル1冊分の書類がテーブルに置かれている様子" \
        --output ../assets/images/doc-amount-sample.png

    python lp-image-create.py --preset icon \
        --prompt "財産と書類を一緒に整理することを表す、シンプルなアイコン" \
        --output ../assets/images/icon-organize.png

環境変数 — nomoto-tax-lp/.env に設定:
    GOOGLE_CLOUD_PROJECT: Vertex AIプロジェクトID
    GOOGLE_CLOUD_LOCATION: リージョン（既定値: global）
    GEMINI_API_KEY: Google Gemini APIキー（Vertex AIを使わない場合）
"""

import argparse
import io
import os
from pathlib import Path
from typing import Optional

from PIL import Image
from google import genai
from google.genai import types


BRAND_DIRECTION = """
【野本裕美 訪問相談LPのブランド方針】
- 基調色は深紺、クリーム／アイボリー。アクセントに柔らかな金
- 「診断」「調査」ではなく「確認」「整理」を行う専門家としての、落ち着いた信頼感を出す
- 明るい自然光、清潔で落ち着いた日本の住空間（訪問相談を想定した自宅のリビング等）
- シニアを一律に白髪、杖、介護が必要な状態として描かず、意思を持つ主体として表現する
- 相続、税金、死、老いを不安をあおる形で扱わない。「争族」を面白がる・煽る表現は厳禁
- 「相続対策で必ず安心できる」といった断定的な演出は避ける
- 家族間の対立や不仲を前提とした構図にしない。前向きな準備の場面として描く
- 財産の多さを前提・誇示する演出（豪邸、宝飾品、札束、高級車等）は避ける
"""


LP_PRESETS = {
    "hero": {
        "name": "ファーストビュー",
        "aspect_ratio": "16:9",
        "description": "訪問相談の安心感と、書類を持って出かけなくてよい気軽さを伝える背景画像",
        "style_hint": """
写実的で自然な写真風の画像にしてください。
- 相談する本人が主役で、税理士が一方的に教える構図にしない
- 人物は中央を避け、見出しを置ける十分な余白を設ける
- 自宅を訪問した落ち着いた雰囲気で、オフィスや会議室のような硬さを出さない
""",
    },
    "consultation": {
        "name": "訪問相談風景",
        "aspect_ratio": "4:3",
        "description": "自宅のテーブルで税金や財産の書類を一緒に確認する様子",
        "style_hint": """
自然な訪問相談風景の写真にしてください。
- 対面で穏やかに話す税理士と本人を、対等な関係として描く
- テーブルには少量の書類、ノート、筆記具を置く（書類の山や乱雑さは避ける）
- 書類の文字、金融機関名、金額は判読できないようにする
- 営業面談や査定のような威圧的な印象を避ける
""",
    },
    "portrait": {
        "name": "担当者紹介（参考素材）",
        "aspect_ratio": "1:1",
        "description": "専門性と親しみやすさを伝える人物写真の参考イメージ",
        "style_hint": """
自然で清潔感のあるポートレート写真にしてください。
- 胸から上、自然な表情、落ち着いた服装
- 背景は白、クリーム、または明るい自宅の一室
- 過剰なビジネス感や堅すぎる演出を避ける
- 実際の担当者紹介（野本裕美本人）には生成画像を使わず、本人写真を使用すること。これはレイアウト確認用の参考素材に限る
""",
    },
    "doc-amount": {
        "name": "書類の目安イメージ",
        "aspect_ratio": "4:3",
        "description": "「A4ファイル1冊または手提げ袋1つ程度」の書類の量感を伝える写真",
        "style_hint": """
書類の分量が直感的に伝わる、生活感のある写真にしてください。
- A4サイズのクリアファイルや書類ファイル1冊分程度の量にする（山積みや大量の書類にしない）
- テーブルや床など、自宅内の自然な置き場所を想定する
- 書類の文字、金融機関名、個人情報は判読できないようにする
- 几帳面に整頓されすぎず、かといって雑然としすぎない「よくある状態」を表現する
""",
    },
    "section-bg": {
        "name": "セクション背景",
        "aspect_ratio": "16:9",
        "description": "文章の可読性を損なわない淡い背景素材",
        "style_hint": """
深紺、クリーム、白、柔らかな金を使った控えめな抽象背景にしてください。
- 装飾は少なく、余白を広く取る
- 上に深紺色の文章を載せても読みやすい明度にする
- 花柄や高齢者向け広告に見える装飾は避ける
""",
    },
    "icon": {
        "name": "サービスアイコン",
        "aspect_ratio": "1:1",
        "description": "税金、財産、書類整理などを示すシンプルなイラスト",
        "style_hint": """
線を生かしたシンプルなアイコン風イラストにしてください。
- 深紺、クリーム、柔らかな金の2〜3色を使用する
- 小さく表示しても意味が分かる単純な形にする
- コイン、札束、盾、ハートなど既視感の強い記号に頼りすぎない
- 同じLP内で並べやすい統一された線幅と余白にする
""",
    },
    "ogp": {
        "name": "OGP画像",
        "aspect_ratio": "16:9",
        "description": "SNSやメッセージで共有した際に表示する画像",
        "style_hint": """
落ち着いた世界観が一目で伝わる画像にしてください。
- 右側または左側にタイトルを重ねられる広い余白を設ける
- 画像内に文字やロゴは生成しない
- 小さな表示でも暗く沈まず、人物やモチーフを詰め込みすぎない
""",
    },
}


LP_STYLES = {
    "navy-cream": "深紺とクリームを中心に、自然光のある落ち着いた雰囲気。親しみやすいが軽すぎない。",
    "trustworthy": "白と深紺を基調にした端正で信頼感のある雰囲気。堅苦しくせず、清潔で落ち着いた構図。",
    "natural-photo": "広告らしく作り込みすぎない、自然な日本の暮らしを感じる写真。現実的な人物と室内。",
    "minimal": "余白を生かし、要素を絞ったミニマルなデザイン。文章の邪魔をしない。",
    "soft-illustration": "淡い色面と細い深紺の線を使った、落ち着いたフラットイラスト。子ども向けにはしない。",
}


def load_env_file() -> None:
    """スクリプト近傍または実行ディレクトリから上方向に.envを探す。"""
    checked = set()
    for start_dir in (Path(__file__).parent, Path.cwd()):
        current = start_dir.resolve()
        while True:
            env_path = current / ".env"
            if env_path not in checked and env_path.is_file():
                print(f".envを読み込み: {env_path}")
                with env_path.open(encoding="utf-8") as env_file:
                    for raw_line in env_file:
                        line = raw_line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, _, value = line.partition("=")
                            os.environ[key.strip()] = value.strip().strip("'\"")
                return
            checked.add(env_path)
            if current == current.parent:
                break
            current = current.parent


def create_client() -> genai.Client:
    """Vertex AIまたはAPIキー方式でクライアントを作成する。"""
    load_env_file()
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if project:
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
        print(f"Vertex AIモード: project={project}, location={location}")
        return genai.Client(vertexai=True, project=project, location=location)

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key:
        print("APIキーモード")
        return genai.Client(api_key=api_key)

    raise EnvironmentError(
        "GOOGLE_CLOUD_PROJECTまたはGEMINI_API_KEYを.envに設定してください"
    )


def build_prompt(user_prompt: str, preset: Optional[str], style: Optional[str]) -> str:
    """利用者の指示に、用途・ブランド・共通制約を加える。"""
    parts = []
    if preset:
        selected_preset = LP_PRESETS[preset]
        parts.append(
            f"【用途】{selected_preset['name']} — {selected_preset['description']}"
        )
        parts.append(selected_preset["style_hint"])

    parts.append(BRAND_DIRECTION)

    if style:
        parts.append(f"【スタイル】{LP_STYLES[style]}")

    parts.append("""
【共通の制約】
- 公認会計士・税理士 野本裕美の訪問相談LPに使用する画像素材
- HTML/CSSで文章を重ねるため、画像内に文字、ロゴ、透かしを入れない
- 公認会計士・税理士の資格や業務範囲（相続税申告、財産評価、法律相談等は範囲外）を画像だけで誤認させない
- 実在の相談者の声に見せるための架空の人物写真として使わない
- Web表示に適した、自然で高品質な仕上がりにする
- 指や手、書類、眼鏡などの形を不自然にしない
""")
    parts.append(f"【生成する画像の内容】\n{user_prompt}")
    return "\n".join(parts)


IMAGE_GENERATION_MODELS = [
    "gemini-3-pro-image",
    "gemini-2.5-flash-image",
]


def generate_with_fallback(client, contents, aspect_ratio: str = "16:9"):
    """高品質モデルから順に画像生成を試す。"""
    last_error = None
    for model_name in IMAGE_GENERATION_MODELS:
        try:
            print(f"モデル使用: {model_name}")
            return client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                    image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
                ),
            )
        except Exception as error:
            print(f"エラー: {model_name} - {type(error).__name__}: {error}")
            last_error = error
    raise RuntimeError(f"すべてのモデルが利用できません。最後のエラー: {last_error}")


def generate_lp_image(
    prompt: str,
    output_path: str,
    preset: Optional[str] = None,
    style: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
    reference: Optional[str] = None,
) -> str:
    """野本裕美 訪問相談LP用画像を生成して保存する。"""
    client = create_client()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    selected_ratio = (
        aspect_ratio
        or (LP_PRESETS[preset]["aspect_ratio"] if preset else None)
        or "16:9"
    )

    if reference:
        reference_path = Path(reference)
        if not reference_path.is_file():
            raise FileNotFoundError(f"参考画像が見つかりません: {reference}")
        print(f"画像編集モード: {reference}")
        ref_image = Image.open(reference_path)
        edit_prompt = build_prompt(
            f"参考画像を基に、次の指示で修正してください。\n{prompt}",
            preset,
            style,
        )
        response = generate_with_fallback(client, [edit_prompt, ref_image], selected_ratio)
    else:
        full_prompt = build_prompt(prompt, preset, style)
        print(
            f"プリセット: {preset or 'なし'} / "
            f"スタイル: {style or 'デフォルト'} / "
            f"アスペクト比: {selected_ratio}"
        )
        response = generate_with_fallback(client, full_prompt, selected_ratio)

    for candidate in response.candidates or []:
        for part in candidate.content.parts or []:
            if part.inline_data is not None:
                image = Image.open(io.BytesIO(part.inline_data.data))
                image.save(output_path)
                print(f"画像を保存しました: {output_path}")
                return output_path

    raise RuntimeError("画像が生成されませんでした。プロンプトを変えて再試行してください")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="野本裕美 訪問相談LP用画像素材をGeminiで生成します"
    )
    parser.add_argument("--prompt", "-p", help="生成する画像の内容")
    parser.add_argument("--output", "-o", help="出力ファイルパス")
    parser.add_argument(
        "--preset",
        choices=list(LP_PRESETS),
        help="用途プリセット",
    )
    parser.add_argument(
        "--style",
        "-s",
        choices=list(LP_STYLES),
        default="navy-cream",
        help="画像スタイル（既定値: navy-cream）",
    )
    parser.add_argument(
        "--aspect-ratio",
        "-a",
        choices=["1:1", "16:9", "9:16", "4:3", "3:4"],
        help="アスペクト比。未指定時はプリセットから決定",
    )
    parser.add_argument("--reference", "-r", help="編集元となる参考画像")
    parser.add_argument("--list-presets", action="store_true", help="一覧を表示")
    args = parser.parse_args()

    if args.list_presets:
        print("\n=== 野本裕美 訪問相談LP用プリセット ===\n")
        for key, preset in LP_PRESETS.items():
            print(
                f"  {key:14} {preset['aspect_ratio']:5}  "
                f"{preset['name']} — {preset['description']}"
            )
        print("\n=== スタイル ===\n")
        for key, description in LP_STYLES.items():
            print(f"  {key:18} {description}")
        return

    if not args.prompt or not args.output:
        parser.error("--promptと--outputは必須です（一覧表示時を除く）")

    generate_lp_image(
        prompt=args.prompt,
        output_path=args.output,
        preset=args.preset,
        style=args.style,
        aspect_ratio=args.aspect_ratio,
        reference=args.reference,
    )


if __name__ == "__main__":
    main()
