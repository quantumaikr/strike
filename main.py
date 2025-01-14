import os
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import cv2
import numpy as np
from pdf2image import convert_from_path
from pathlib import Path
from datetime import datetime

app = FastAPI()

# Static 및 템플릿 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# static 폴더 내 이미지 파일 정리 함수
def cleanup_static_folder():
    """static 폴더 내의 모든 PDF와 이미지 파일을 삭제"""
    static_dir = Path("static")
    if static_dir.exists():
        for file in static_dir.glob("*"):
            if file.suffix.lower() in ['.pdf', '.png', '.jpg', '.jpeg']:
                try:
                    file.unlink()
                except Exception as e:
                    print(f"파일 삭제 중 오류 발생: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # 파일 업로드 시작 시 이전 파일들 정리
        cleanup_static_folder()
        
        # static 폴더가 없으면 생성
        os.makedirs("static", exist_ok=True)

        # 업로드된 파일 저장
        upload_path = f"static/{file.filename}"
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # PDF를 이미지로 변환
        images = convert_from_path(upload_path, dpi=300, output_folder="static", fmt="png")
        processed_images = []

        for i, image in enumerate(images):
            # 원본 이미지 저장
            original_path = f"static/original_page_{i+1}.png"
            image.save(original_path)

            # 취소선 제거
            processed_path = f"static/processed_page_{i+1}.png"
            process_image(original_path, processed_path)
            processed_images.append((original_path, processed_path))

        # 업로드된 PDF 파일 삭제
        Path(upload_path).unlink()

        # 타임스탬프 생성
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # 결과 HTML 생성
        result_html = '<div class="pages-container">'
        
        for i, (original_path, processed_path) in enumerate(processed_images, 1):
            result_html += f'''
            <div class="page-section">
                <h2>페이지 {i}</h2>
                <div class="image-comparison">
                    <div class="image-container">
                        <h3>원본 이미지</h3>
                        <img src="/{original_path}?t={timestamp}" alt="원본 이미지 {i}">
                    </div>
                    <div class="image-container">
                        <h3>처리된 이미지</h3>
                        <img src="/{processed_path}?t={timestamp}" alt="처리된 이미지 {i}">
                    </div>
                </div>
                <div class="download-section">
                    <a href="/{processed_path}?t={timestamp}" download class="download-button">처리된 이미지 다운로드</a>
                </div>
            </div>'''
        
        result_html += '</div>'
        
        return HTMLResponse(content=result_html)
        
    except Exception as e:
        return HTMLResponse(content=f'<div class="error">처리 중 오류가 발생했습니다: {str(e)}</div>')

# 취소선 제거 함수
def process_image(original_path, output_path):
    # 이미지 읽기
    img = cv2.imread(original_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 빨간색 마스크 범위 설정
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # 빨간색 마스크 생성
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.add(mask1, mask2)

    # 컨투어 검출 및 흰색으로 채우기
    red_lines = cv2.bitwise_and(img, img, mask=mask)
    gray = cv2.cvtColor(red_lines, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x - 2, y - 18), (x + w + 2, y + h + 18), (255, 255, 255), -1)

    # 결과 이미지 저장
    cv2.imwrite(output_path, img)
