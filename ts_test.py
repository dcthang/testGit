import trans_probs as t
import os


def scan_dir(path):

    for filename in os.listdir(path):

        file_path = os.path.join(path, filename);

        if os.path.isdir(file_path):
            print("scan dir " + file_path)
            scan_dir(file_path)
            continue

        if not filename.lower().endswith(".jpg"):
            # print("ignored file " + filename)
            continue

        print("process file "+filename)

        rs = t.predict_final(file_path)

        tex_path = os.path.join(path, filename.lower().replace(".jpg",".txt"))
        with open(tex_path, 'w') as f:
            f.write(rs)


if __name__ == '__main__':
#   scan_dir("/Users/thanhtruongle/Desktop/20200724_datatest")
    scan_dir("/home/dcthang/Projects/MathFormulaReg/Docs/Test/20200724_datatest/A50/A1")
    
