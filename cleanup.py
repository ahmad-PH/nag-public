import sys
import os
import re 

class TempChangeDir(object):
    def __init__(self, target_dir):
        self.target_dir = target_dir

    def __enter__(self):
        self.initial_dir = os.getcwd()
        os.chdir(self.target_dir)

    def __exit__(self, exec_type, exec_value, exec_traceback):
        os.chdir(self.initial_dir)


def cleanup_models_folder(models_folder):
    os.chdir(models_folder)
    for folder in os.listdir('.'):
        if os.path.isdir(folder):
            if folder.startswith('investigate'):
                if os.path.isfile(folder + '/cleanignore'): continue
                model_name = re.search('investigate_(.+)_.*', folder).group(1)
                for file in os.listdir(folder):
                    cleanup_model_folder(folder + '/' + file, model_name)
            else:
                cleanup_model_folder(folder, folder)
                
def cleanup_model_folder(folder, model_name):
    if os.path.isfile(folder + '/cleanignore'): return

    def create_clean_ignore_file():
        clean_ignore = open('cleanignore', 'x'); clean_ignore.close()

    with TempChangeDir(folder):
        print('cleaning up folder: ', folder)

        indexed_matching_files = []
        for file in os.listdir('.'):
            match = re.search(model_name + '_(\d+).pth', file)
            if match is not None:
                index = int(match.group(1))
                indexed_matching_files.append((index, file))

        if len(indexed_matching_files) == 0: 
            print('no file found with the pattern \"model_name_index.pth\", ignoring the directory.')
            create_clean_ignore_file()
            return

        indexed_matching_files.sort(key = lambda x: x[0], reverse = True)
        last_file = indexed_matching_files[0][1]

        # print('all_matching: ', indexed_matching_files)

        keep_files = [
            last_file,
            "{}-best.pth".format(model_name)
        ]

        if all([file in os.listdir('.') for file in keep_files]) and len(keep_files) == len(os.listdir('.')):
            print('folder only contains last and best models, exiting.')
            create_clean_ignore_file()
            return

        with open('keep.txt', 'w') as out_file:
            for keep_file in keep_files:
                out_file.write(keep_file + '\n')

        print(('files that will be kept are: {}\n' \
              'change the keep.txt if they are wrong, then enter \'y\' to delete the rest.').format(keep_files))

        if input() != 'y':
            print('exiting.')
            sys.exit(0)

        while True:
            with open('keep.txt', 'r') as in_file:
                keep_files = [line.strip() for line in in_file.readlines()]
            everything_ok = True
            for file in keep_files:
                if not os.path.isfile(file):
                    print(('new file in keep_file: \'{}\' doesnt exist.'
                           ' fix it and then press enter to continue.').format(file)) 
                    everything_ok = False
            if everything_ok:
                break
            else:
                input()


        for file in os.listdir('.'):
            if file not in keep_files:
                os.remove(file)

        create_clean_ignore_file()
        print('done.')
