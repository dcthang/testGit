import os

def process_directory(root):

    for item in os.listdir(root):
        if os.path.isdir(item):
            print("is directory", item)
            process_directory(item)
        else:
            path=os.path.dirname(item)
	    folder_name = os.path.basename(path)
	    filename = os.path.splitext(folder_name)[0] 	
	    os.path.join(path,             
            os.rename(item, )

process_directory(os.getcwd())
