
from FlagEmbedding import FlagModel
from pickle import dump as pkl_dump
from argparse import ArgumentParser
from datetime import datetime
from tqdm import tqdm

model = FlagModel('BAAI/bge-base-en-v1.5')

def load_data(data_fl):

    '''
    Load data from the file into a list of conversations
    
    Args:
    data_fl: str, path to the file containing the data (txt file)
    
    Returns:
    gat_data_lst: list, list of conversations
    '''

    with open(data_fl, 'rb') as f:
        lines = f.readlines()

    # replace '\r\n' with '\n' and join all lines
    gat_data = ''.join([line.decode('utf-8').replace('\r\n', '\n') for line in lines])
    return gat_data.split('\n\n')

def generate_embeddings(data_lst):

    '''
    Generate embeddings for the list of conversations

    Args:
    data_lst: list, list of conversations

    Returns:
    embeddings: list, list of embeddings
    '''

    embeddings = []
    for d in tqdm(data_lst):
        if d == '':
            continue
        d = d.replace('\n', ' ')
        embeddings.append(model.encode(d))
    return embeddings

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--data_fl', type=str, help='Path to the file containing the data (txt file)', default='data_generation/gat_refined.txt')
    args = parser.parse_args()
    embeddings_fl = 'gat_embeddings.pkl'

    # load data
    start1 = datetime.now()
    print(f"\n{'='*50}\nLoading data from {args.data_fl}...")
    gat_data_lst = load_data(args.data_fl)
    print('Data loaded successfully')

    # generate embeddings
    print('\nGenerating embeddings...')
    embeddings = generate_embeddings(gat_data_lst)
    print('Embeddings generated successfully')

    # save embeddings as pkl file
    with open(embeddings_fl, 'wb') as f:
        pkl_dump(embeddings, f)

    print(f'\nSuccessfully stored embeddings into {embeddings_fl} in {(datetime.now() - start1).seconds} seconds')
    print('='*50, '\n')