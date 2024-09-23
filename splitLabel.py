import os
import pandas as pd

datasetDir = "/home/actvn/KeyBoard/pythonProject/.venv/wav"
trainDir = "/home/actvn/KeyBoard/pythonProject/.venv/train"
valDir = "/home/actvn/KeyBoard/pythonProject/.venv/val"
testDir = "/home/actvn/KeyBoard/pythonProject/.venv/test"
# Danh sách để lưu thông tin của các tệp
trainData = []
for folder_name in os.listdir(trainDir):
    files = os.path.join(trainDir, folder_name)
    if files.endswith('.wav'):
            # Lấy tên tệp
        file_name = os.path.basename(files)
            # Lấy phần của tên tệp mà không có phần mở rộng
        file_name_without_extension = os.path.splitext(file_name)[0]
            # Lấy ký tự đầu tiên của tên tệp
        first_character = file_name_without_extension[0]
            # Tạo một bản ghi cho tệp hiện tại và thêm vào danh sách dữ liệu
        trainData.append({'slice_file_name': file_name, 'class': first_character})

# Tạo DataFrame từ danh sách dữ liệu
df = pd.DataFrame(trainData)
# Ghi DataFrame vào tệp CSV
df.to_csv('train_dataset.csv', index=False)
# Lưu DataFrame vào tệp CSV
df.to_csv('/home/actvn/KeyBoard/pythonProject/.venv/train_dataset.csv', index=False)
# Hiển thị nội dung của DataFrame
print(df)

testData = []
for folder_name in os.listdir(testDir):
    files = os.path.join(testDir, folder_name)
    if files.endswith('.wav'):
            # Lấy tên tệp
        file_name = os.path.basename(files)
            # Lấy phần của tên tệp mà không có phần mở rộng
        file_name_without_extension = os.path.splitext(file_name)[0]
            # Lấy ký tự đầu tiên của tên tệp
        first_character = file_name_without_extension[0]
            # Tạo một bản ghi cho tệp hiện tại và thêm vào danh sách dữ liệu
        testData.append({'slice_file_name': file_name, 'class': first_character})

# Tạo DataFrame từ danh sách dữ liệu
df_test = pd.DataFrame(testData)
# Ghi DataFrame vào tệp CSV
df_test.to_csv('test_dataset.csv', index=False)
# Lưu DataFrame vào tệp CSV
df_test.to_csv('/home/actvn/KeyBoard/pythonProject/.venv/test_dataset.csv', index=False)
# Hiển thị nội dung của DataFrame
print(df_test)

valData = []
for folder_name in os.listdir(valDir):
    files = os.path.join(valDir, folder_name)
    if files.endswith('.wav'):
            # Lấy tên tệp
        file_name = os.path.basename(files)
            # Lấy phần của tên tệp mà không có phần mở rộng
        file_name_without_extension = os.path.splitext(file_name)[0]
            # Lấy ký tự đầu tiên của tên tệp
        first_character = file_name_without_extension[0]
            # Tạo một bản ghi cho tệp hiện tại và thêm vào danh sách dữ liệu
        valData.append({'slice_file_name': file_name, 'class': first_character})

# Tạo DataFrame từ danh sách dữ liệu
df_val = pd.DataFrame(valData)
# Ghi DataFrame vào tệp CSV
df_val.to_csv('val_dataset.csv', index=False)
# Lưu DataFrame vào tệp CSV
df_val.to_csv('/home/actvn/KeyBoard/pythonProject/.venv/val_dataset.csv', index=False)
# Hiển thị nội dung của DataFrame
print(df_val)