import shutil

old_file = r'C:\Users\鞠新宇\Pictures\1000\fishBig-1.jpg'
for i in range(50):
    n = str(i + 1).zfill(3)
    new_file = r'C:\Users\鞠新宇\Pictures\1000\fishBig' + n + '.jpg'
    print(new_file)
    shutil.copyfile(old_file, new_file)
