import os, shutil
import pickle
import subprocess
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

TEMP_DIR = "temp"
OUT_DIR = "out"
IMG_DIR = f"{OUT_DIR}/img"
FILE_PATH = "syscalls"
SYSCALLS = ["snd-cert", "snd-unm"]
LABELS = ["self","nonself"]

def create_self_nonself_files():
    """
    Splits the original snd-cert/unm.x.test files into self and non-self files
    """
    for syscall_type in SYSCALLS:
        self_file = open(f"{TEMP_DIR}/{syscall_type}-self.test", "w")
        nonself_file = open(f"{TEMP_DIR}/{syscall_type}-nonself.test", "w")
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
        syscalls_split_file = open(f"{TEMP_DIR}/{syscall_type}-split.train", "w")
        snd_train_path = f"{FILE_PATH}/{syscall_type}/{syscall_type}.train"
        with open(snd_train_path) as train_file:
            for syscall in train_file:
                # Generate all n-grams of the current syscall
                n_grams = extract_n_grams(syscall.strip(),chunk_size,unique=True)
                if len(n_grams)==0:
                    continue
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
            filename = f"{TEMP_DIR}/{syscall_type}-{label}"
            # Create file for the syscall chunks
            syscalls_split_file = open(filename+"-split.test", "w")
            with open(filename+".test") as syscalls_file:
                for syscall in syscalls_file:
                    # Generate all n-grams of the current syscall
                    n_grams = extract_n_grams(syscall.strip(),chunk_size,unique=False)
                    if len(n_grams)==0:
                        continue
                    # Write n-grams to syscall chunks file
                    syscalls_split_file.writelines(n_grams)
                    # Keep track of end position in chunks file of current syscall
                    syscall_pos += len(n_grams)
                    syscall_positions.append(syscall_pos)
            syscall_type_dict[f"{syscall_type}-{label}"] = syscall_positions
            syscalls_split_file.close()
    return syscall_type_dict


def extract_n_grams(string, n, overlap=0, unique=True):
    """
    Extracts n-grams from the current string
    NOTE: when unique=True, it only extracts unique n-grams in the current string.
          There might still be duplicates in the collection
    """
    n_grams = []
    step = n - overlap
    n_grams = [string[i:i+n] for i in range(0, len(string)-step+1, step)]

    if unique:
        n_grams = list(set(n_grams))

    n_grams = [g+"\n" for g in n_grams]

    return n_grams


def merge_test_files():
    """
    Merges the self and nonself .test files created by split_test_into_chunks
    """
    for syscall_type in SYSCALLS:
        self_file = open(f"{TEMP_DIR}/{syscall_type}-self-split.test")
        nonself_file = open(f"{TEMP_DIR}/{syscall_type}-nonself-split.test")
        merged_file = open(f"{TEMP_DIR}/{syscall_type}-merged-split.test", "w")
        merged_lines = self_file.readlines()
        merged_lines.extend(nonself_file.readlines())
        merged_file.writelines(merged_lines)
        self_file.close()
        nonself_file.close()
        merged_file.close()


def negative_selection(syscall_type, n, r, count=True, log=False):
    alphabet_file = f"file://{FILE_PATH}/{syscall_type}/{syscall_type}.alpha"
    self_file = f"{TEMP_DIR}/{syscall_type}-split.train"
    count_param = "-c" if count else ""
    log_param = "-l" if log else ""
    os.system(f"java -jar negsel2.jar -alphabet {alphabet_file} -self {self_file} -n {n} -r {r} \
                {count_param} {log_param} < {TEMP_DIR}/{syscall_type}-merged-split.test \
                > {OUT_DIR}/N={n}-R={r}-{syscall_type}-output")


def collect_predictions(n, r, syscall_type, syscall_type_dict):
    output_file = open(f"{OUT_DIR}//N={n}-R={r}-{syscall_type}-output", "r")
    indexes = syscall_type_dict[f"{syscall_type}-self"]
    nonself_indexes = syscall_type_dict[f"{syscall_type}-nonself"]
    nonself_indexes = np.array(nonself_indexes) + indexes[-1]
    indexes.extend(nonself_indexes.tolist())
    pred = []
    start = 0
    for i in indexes:
        # Read n lines from the output file
        results = [float(next(output_file).strip()) for _ in range(i-start)]
        total_anomaly = np.mean(results)
        pred.append(total_anomaly)
        start = i

    return pred

def generate_roc(n, r, ground, pred):
    fpr, tpr, _ = roc_curve(ground, pred)
    auc_score = auc(fpr, tpr)
    ns_fpr, ns_tpr, _ = roc_curve(ground, np.zeros(len(pred),))
    plt.figure(figsize=(7, 7))
    plt.grid()
    plt.plot(fpr, tpr, marker='.', label=f'Negative Selection (AUC={auc_score:.3f})', linewidth=2)
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Chance', linewidth=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"ROC-curve for Negative Selection with N={n} and R={r}")
    plt.legend(loc=4, framealpha=1)

def run_negative_selection(n, r):
    # Preparation
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    if not os.path.exists(OUT_DIR) or not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)

    print("Preparing files")
    create_self_nonself_files()
    syscall_type_dict = split_test_into_chunks(n)
    # Write syscall_type_dict for reproducibility
    with open(f"{OUT_DIR}/N={n}-R={r}-indices.pkl", 'wb') as f:
        pickle.dump(syscall_type_dict, f, pickle.HIGHEST_PROTOCOL)
    split_train_into_chunks(n)
    merge_test_files()

    for syscall_type in SYSCALLS:
        print(f"Negative selection for {syscall_type} (may take several minutes)")
        negative_selection(syscall_type, n, r, count=True, log=True)

    print("Calculate results")
    for syscall_type in SYSCALLS:
        pred = collect_predictions(n, r, syscall_type, syscall_type_dict)
        plt.figure(figsize=(8,8))
        plt.hist(pred)
        plt.savefig(f"{IMG_DIR}/N={n}-R={r}-{syscall_type}-hist.pdf")
        plt.savefig(f"{IMG_DIR}/N={n}-R={r}-{syscall_type}-hist.png")
        # Ground truth predictions
        ground = np.zeros(len(pred))
        num_of_nonself = len(syscall_type_dict[f"{syscall_type}-nonself"])
        ground[-num_of_nonself:] = 1
        # Generate ROC plots
        generate_roc(n,r, ground, pred)
        plt.savefig(f"{IMG_DIR}/N={n}-R={r}-{syscall_type}-roc.pdf")
        plt.savefig(f"{IMG_DIR}/N={n}-R={r}-{syscall_type}-roc.png")
    
    # Clean-up
    shutil.rmtree(TEMP_DIR)

    print("Done")

def main():
    n_list = [3,5,7]
    r_list = [2,4,6]

    for n, r in zip(n_list, r_list):
        print(f"Running negsel with N={n} and R={r}")
        run_negative_selection(n, r)

if __name__ == "__main__":
    main()
