import pandas as pd
data = {
    'Tên': ['An', 'Bình', 'Chi', 'Dũng', 'Em'],
    'Số_Giờ_Học': [1, 3, 5, 8, 10],
    'Điểm_Thi': [2, 5, 6, 9, 10]
}
df = pd.DataFrame(data)

print(df)
print(df.describe())

print(df[df['Số_Giờ_Học'] > 4])