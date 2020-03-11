import os
import sys
from shutil import copyfile
import argparse

def arguments():
    '''Gets the command line arguments'''
    parser = argparse.ArgumentParser(description='Prepare input and output for negative selection.')
    parser.add_argument('folder', metavar='folder', type=str,
                    help='folder containing all data')
    parser.add_argument('-c', type=int, required=False, metavar='chunk_size',
                        default=10,
                        help='size of chunks')
    parser.add_argument('-o', type=int, required=False, metavar='overlap',
                        default=0,
                        help='overlap between chunks')

    args = parser.parse_args()
    return args

def split_to_chunks(data, chunk_size=10, overlap=0):
    '''Splits a string into equal sized chunks'''
    step = chunk_size - overlap
    return [data[i:i+chunk_size] for i in range(0, len(data)-step+1, step)]

def preprocess_train(filename, chunk_size, overlap):
    '''Splits the trainset into chunks'''
    input = open(filename, 'r')
    out = open('train', 'w')
    for l in input:
        l = l[:-1] # ignore newline
        chunks = split_to_chunks(l, chunk_size, overlap)
        chunks = '\n'.join(chunks)
        out.write(chunks)
        out.write('\n')

    input.close()
    out.close()

def preprocess_test(filename, labels, chunk_size, overlap):
    '''Splits the test set into chunks and stores them into
       two files: one for normal data and one for anomalous'''

    input = open(filename, 'r')
    labels = open(labels, 'r')
    out_normal = open('test.normal', 'w')
    out_anomalous = open('test.anomalous', 'w')

    for l in input:
        l = l[:-1] # ignore newline
        c = next(labels)[:-1] # get label of data
        chunks = split_to_chunks(l[:-1], chunk_size, overlap)
        chunks = '\n'.join(chunks)
        if c == '0':
            out_normal.write(chunks)
            out_normal.write('\n')
        else:
            out_anomalous.write(chunks)
            out_anomalous.write('\n')

    input.close()
    labels.close()
    out_normal.close()
    out_anomalous.close()

def main():
    args = arguments()
    folder = args.folder
    dataset_name = os.path.basename(os.path.normpath(folder))
    train_filename = os.path.join(folder, dataset_name+'.train')
    test_filename = os.path.join(folder, dataset_name+'.1.test')
    labels_filename = os.path.join(folder, dataset_name+'.1.labels')
    alphabet_filename = os.path.join(folder, dataset_name+'.alpha')

    # Copy the alphabet file into this folder
    copyfile(alphabet_filename, 'alphabet')
    # Preprocess the train and test files
    preprocess_train(train_filename, args.c, args.o)
    preprocess_test(test_filename, labels_filename, args.c, args.o)

if __name__ == "__main__":
    main()
