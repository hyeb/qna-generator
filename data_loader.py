
FILE_PATH = 'file path'

def load_aihub_data(file_path: str):
    """
    file path의 파일 로드 후 해당 aihub의 데이터 구조 내 'original_text' 키의 값을 Context로 활용 예정

    args: 로드할 파일 경로
    returns: 추출된 original_text가 들어간 리스트
    """

    try:
        with open(FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

    except FileNotFoundError:
        print(f"파일 {FILE_PATH}을 찾을 수 없습니다. 경로를 다시 확인해주세요")
        return []
    

    try:
        section_info = data['training_data_info']['section_info']
        contents = [section_info['original_text'] for section in section_info]
        return contents
    except:
        print(f"Contetns를 찾을 수 없습니다")
        return []