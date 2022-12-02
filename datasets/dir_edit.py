import os
import shutil
import sys
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--dest-dir",type=str)
arg_parser.add_argument("--new-file",type=str)
arg_parser.add_argument("--old-file",type=str)
arg_parser.add_argument("--parent-store-dir",type=str)

args = arg_parser.parse_args()

shutil.copyfile(args.new_file,os.path.join(args.dest_dir,args.new_file))
#shutil.move(args.new_file,os.path.join(args.dest_dir,args.new_file))


if args.new_file in os.listdir(args.dest_dir): #check if move succesful
    os.remove(os.path.join(args.dest_dir,args.old_file))
else:
    raise Exception("Replacement file not found in destination directory. Aborting deletion")

os.rename(os.path.join(args.dest_dir,args.new_file),os.path.join(args.dest_dir,args.old_file))

if os.path.isdir(args.parent_store_dir):
    shutil.rmtree(args.parent_store_dir)


os.mkdir(args.parent_store_dir)
shutil.copyfile(args.new_file,os.path.join(args.parent_store_dir,args.new_file))

