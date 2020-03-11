import os
import subprocess

FILE_PATH = "syscalls"
SYSCALLS = ["snd-cert", "snd-unm"]
LABELS = ["self","nonself"]

def create_self_nonself_files():
    """
    Splits the original snd-cert/unm.x.test files into self and non-self files
    """
    for syscall_type in SYSCALLS:
        self_file = open(f"{syscall_type}-self.test", "w")
        nonself_file = open(f"{syscall_type}-nonself.test", "w")
        snd_path = f"{FILE_PATH}/{syscall_type}/{syscall_type}"
        for file_nr in range(1,4):
            with open(f"{snd_path}.{file_nr}.test") as test_file, open(f"{snd_path}.{file_nr}.labels") as labels_file:
                for syscall, lbl in zip(test_file, labels_file):
                    syscall = syscall.strip(); lbl = lbl.strip()
                    if int(lbl) == 1:
                        nonself_file.write(syscall+"\n")
                    else:
                        self_file.write(syscall+"\n")
        self_file.close()
        nonself_file.close()


def split_train_into_chunks(chunk_size):
    """
    Splits the train files into chunks
    """
    for syscall_type in SYSCALLS:
        syscalls_split_file = open(f"{syscall_type}-split.train", "w")
        snd_train_path = f"{FILE_PATH}/{syscall_type}/{syscall_type}.train"
        with open(snd_train_path) as train_file:
            for syscall in train_file:
                # Generate all n-grams of the current syscall
                n_grams = extract_unique_n_grams(syscall,chunk_size)
                # Write n-grams to syscall chunks file
                syscalls_split_file.writelines(n_grams)
        syscalls_split_file.close()


def split_test_into_chunks(chunk_size):
    """
    Splits the test files created from create_self_nonself_files into chunks
    Also maintains a dictionary with the starting position of every syscall for every file
    """
    syscall_type_dict = dict()
    # Loop over all files
    for syscall_type in SYSCALLS:
        for label in LABELS:
            syscall_positions = []
            syscall_pos = 0
            filename = f"{syscall_type}-{label}"
            # Create file for the syscall chunks
            syscalls_split_file = open(filename+"-split.test", "w")
            with open(filename+".test") as syscalls_file:
                for syscall in syscalls_file:
                    # Generate all n-grams of the current syscall
                    n_grams = extract_unique_n_grams(syscall,chunk_size)
                    # Write n-grams to syscall chunks file
                    syscalls_split_file.writelines(n_grams)
                    # Keep track of end position in chunks file of current syscall
                    syscall_pos += len(n_grams)
                    syscall_positions.append(syscall_pos)
            syscall_type_dict[filename] = syscall_positions
            syscalls_split_file.close()
    return syscall_type_dict


def extract_unique_n_grams(string, n):
    """
    Extracts unique n-grams from the current string
    NOTE: it only extracts unique n-grams in the current string, there might still be duplicates in the collection
    """
    n_grams = []
    for i in range(len(string)-n+1):
        n_gram = string[i:i+n]
        if n_gram not in n_grams:
            n_grams.append(n_gram+"\n")
    return n_grams


def merge_test_files():
    """
    Merges the self and nonself .test files created by split_test_into_chunks
    """
    for syscall_type in SYSCALLS:
        self_file = open(f"{syscall_type}-self-split.test")
        nonself_file = open(f"{syscall_type}-nonself-split.test")
        merged_file = open(f"{syscall_type}-merged-split.test", "w")
        merged_lines = self_file.readlines()
        merged_lines.extend(nonself_file.readlines())
        merged_file.writelines(merged_lines)
        self_file.close()
        nonself_file.close()
        merged_file.close()


def negative_selection(n, r, count=True, log=False):
    """
    TODO
    """
    # alphabet_file = f"file://{FILE_PATH}/{syscall_type}/{syscall_type}.alpha"
    # self_file = f"{syscall_type}.train"
    # os.system(f"java -jar negsel2.jar -alphabet {alphabet_file} -self {self_file} -n {n} -r {r} < merged > output")


def main():
    CHUNK_SIZE = 7
    create_self_nonself_files()
    syscall_type_dict = split_test_into_chunks(CHUNK_SIZE)
    split_train_into_chunks(CHUNK_SIZE)
    merge_test_files()

    negative_selection(CHUNK_SIZE, 4, count=True, log=False)


if __name__ == "__main__":
    main()