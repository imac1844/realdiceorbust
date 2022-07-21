import os
for root, dirs, files in os.walk("./DiceSets/", topdown=True):
   for name in files:
      print(os.path.join(root, name))
