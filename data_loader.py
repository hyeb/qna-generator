import os
import json


def get_text_contents(folder_path):
    text_list = []

    if not os.path.isdir(folder_path):
        print(f"{folder_path}를 찾을 수 없음")
        return
    
    json_list = os.listdir(folder_path)

    for j_file in json_list:
        if j_file.endswith('.json'):
            json_path = os.path.join(folder_path, j_file)

            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    text = json.load(f)

                text_content = text.get("06_transcription", {}).get("1_text")

                if text_content:
                    text_list.append(text_content)
                else:
                    print(f"{j_file}에서 '1_text'를 찾을 수 없음")
            
            except json.JSONDecodeError:
                print(f"{j_file}은 유효한 json 형식이 아님")
            except Exception as e:
                print(f"{j_file} 처리 중 오류 발생: {e}")
    
    comb_text = " ".join(text_list)
    
    return comb_text