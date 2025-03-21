import csv
from google.oauth2 import service_account
from googleapiclient.discovery import build


def load_coordinates_from_doc(doc_url):
    """从Google Doc读取坐标数据，解析为(x, y, char)元组列表"""
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    creds = service_account.Credentials.from_service_account_file('credentials.json', scopes=SCOPES)
    service = build('docs', 'v1', credentials=creds)
    doc = service.documents().get(documentId=doc_url.split('/')[-1]).execute()

    coords = []
    for elem in doc['body']['content']:
        if 'table' in elem:
            for row in elem['table']['tableRows']:
                cells = row['tableCells']
                x = int(cells[0]['content'][0]['paragraph']['elements'][0]['textRun']['content'].strip())
                char = cells[1]['content'][0]['paragraph']['elements'][0]['textRun']['content'].strip()
                y = int(cells[2]['content'][0]['paragraph']['elements'][0]['textRun']['content'].strip())
                coords.append((x, y, char))
    return coords


def create_character_grid(coords):
    max_x = max(c[0] for c in coords) if coords else 0
    max_y = max(c[1] for c in coords) if coords else 0
    grid = [[" " for _ in range(max_x + 1)] for _ in range(max_y + 1)]
    for x, y, char in coords:
        grid[y][x] = char
    grid_rows = ["".join(row) for row in reversed(grid)]  # 修正y轴方向
    return grid_rows


def print_grid(grid_rows):
    for row in grid_rows:
        print(row)


def main(doc_url):
    coords = load_coordinates_from_doc(doc_url)
    grid = create_character_grid(coords)
    print_grid(grid)


if __name__ == "__main__":
    main("https://docs.google.com/document/d/e/2PACX-1vQGUck9HIFCyezsrBSnmENk5ieJuYwpt7YHYEzeNJkIb9OSDdx-ov2nRNReKQyey-cwJOoEKUhLmN9z/pub")