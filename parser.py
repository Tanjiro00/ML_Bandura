import PyPDF2
import os
from pdfminer.high_level import extract_pages, extract_text
import pdfplumber
from pdfminer.layout import LTTextContainer, LTChar, LTRect, LTFigure


def text_extraction(element):
    line_text = element.get_text()

    line_formats = []
    for text_line in element:
        if isinstance(text_line, LTTextContainer):
            for character in text_line:
                if isinstance(character, LTChar):
                    line_formats.append(character.fontname)
                    line_formats.append(character.size)
    format_per_line = list(set(line_formats))

    return (line_text, format_per_line)

def extract_table(pdf_path, page_num, table_num):
    pdf = pdfplumber.open(pdf_path)
    table_page = pdf.pages[page_num]
    table = table_page.extract_tables()[table_num]
    return table

def table_converter(table):
    table_string = ''
    for row_num in range(len(table)):
        row = table[row_num]
        cleaned_row = [item.replace('\n', ' ') if item is not None and '\n' in item else 'None' if item is None else item for item in row]
        table_string+=('|'+'|'.join(cleaned_row)+'|'+'\n')

    table_string = table_string[:-1]
    return table_string

def read_pdf_to_docs(pdf_path: str) -> list[dict]:
    """
    Извлекает текст из PDF по страницам и возвращает список словарей,
    где каждая запись содержит текст страницы и информацию об источнике.
    """
    docs = []
    pdf_name = os.path.basename(pdf_path)

    pdfFileObj = open(pdf_path, 'rb')

    pdfReaded = PyPDF2.PdfReader(pdfFileObj)

    output_root = '/content/extracted'

    pdf_dir = os.path.join(output_root, pdf_name)
    os.makedirs(pdf_dir, exist_ok=True)
    print(f"Обрабатывается: {pdf_name} (подкаталог: {pdf_dir})")
    pdf_dir += "/"

    text_per_page = {}
    img_count = 0

    for page_num, page in enumerate(extract_pages(pdf_path)):
        pageObj = pdfReaded.pages[page_num]
        page_text = []
        line_format = []
        text_from_images = []
        text_from_tables = []
        page_content = []
        meta = []

        table_num = 0
        first_element= True
        table_extraction_flag= False
        pdf = pdfplumber.open(pdf_path)
        page_tables = pdf.pages[page_num]
        tables = page_tables.find_tables()

        page_elements = [(element.y1, element) for element in page._objs]
        page_elements.sort(key=lambda a: a[0], reverse=True)

        for i, component in enumerate(page_elements):
            pos = component[0]
            element = component[1]

            if isinstance(element, LTTextContainer):
                if table_extraction_flag == False:
                    (line_text, format_per_line) = text_extraction(element)
                    page_text.append(line_text)
                    line_format.append(format_per_line)
                    page_content.append(str(line_text))
                else:
                    pass

            if isinstance(element, LTFigure):
                # crop_image(element, pageObj, img_count, pdf_dir)
                # convert_to_images(f'{pdf_dir}cropped_image_{img_count}.pdf', img_count, pdf_dir)
                # image_text = image_to_text(f'{pdf_dir}PDF_image_{img_count}.png')
                # text_from_images.append(image_text)
                # page_content.append(f'{pdf_dir}PDF_image_{img_count}.png')
                # page_content.append(str(image_text))
                # meta.append(f'{pdf_dir}PDF_image_{img_count}.png')
                img_count += 1

            if isinstance(element, LTRect):
                if first_element == True and (table_num+1) <= len(tables):
                    lower_side = page.bbox[3] - tables[table_num].bbox[3]
                    upper_side = element.y1
                    table = extract_table(pdf_path, page_num, table_num)
                    table_string = table_converter(table)
                    text_from_tables.append(table_string)
                    page_content.append(str(table_string))
                    table_extraction_flag = True
                    first_element = False
                    page_text.append('table')
                    meta.append('table')

        source = f"{pdf_name} - page {page_num}"
        docs.append({"text": "".join(page_text), "source": source, "content": "".join(page_content), "meta": "".join(meta)})
        print(source)

    pdfFileObj.close()
    return docs