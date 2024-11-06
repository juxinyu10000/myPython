import os
from win32com import client as wc

path = "D:\\8-1\\"   #待处理目录
dest_path = "D:\\8-2\\"
files = []
point = wc.Dispatch('PowerPoint.Application') # 打开应用

"""

for root,dirs,files in os.walk(path):
    print ("root:%s"%(root))
    print ("dirs:%s"%(dirs))
    print ("files:%s"%(files))

"""

for file in os.listdir(path):
    (file_path, temp_file_name) = os.path.split(file)
    (short_name, extension) = os.path.splitext(temp_file_name)
    if file.endswith(".pptx"):
        pptx = point.Presentations.Open(path+file)
        pptx.SaveAs(dest_path + short_name + ".ppt")
        print("另存成功：" + dest_path + short_name + ".ppt")
        pptx.Close()
point.Quit()