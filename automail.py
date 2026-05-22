import os
import pandas as pd  # if you are generating attendance files

# Example: create a dummy attendance sheet
data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Date": ["2026-02-19", "2026-02-19", "2026-02-19"],
    "Status": ["Present", "Absent", "Present"]
}

df = pd.DataFrame(data)

# Save in attendance folder
folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attendance")
os.makedirs(folder, exist_ok=True)

filename = os.path.join(folder, "attendance.xlsx")
df.to_excel(filename, index=False)

print(f"Attendance sheet saved at {filename}")
