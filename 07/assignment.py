from gensim.models import Word2Vec
import pandas as pd

def main():
    tags = pd.read_csv('./hetrec2011-delicious-2k/tags.dat', sep='\t', header=0, engine='python')
    bookmarks = pd.read_csv('./hetrec2011-delicious-2k/bookmarks.dat', sep='\t', header=0, engine='python')
    bookmarks.drop('md5', axis=1, inplace=True)
    bookmarks.drop('md5Principal', axis=1, inplace=True)
    bookmarks.drop('urlPrincipal', axis=1, inplace=True)
    bookmark_tags = pd.read_csv('./hetrec2011-delicious-2k/bookmark_tags.dat', sep='\t', header=0, engine='python')
    bookmark_tags.drop('tagWeight', axis=1, inplace=True)
    tags.columns = ['tagID', 'value']
    bookmarks.columns = ['bookmarkID', 'title', 'url']
    df = pd.merge(pd.merge(bookmarks, bookmark_tags, on=['bookmarkID']), tags, on=['tagID'])
    
    bm = df.groupby('bookmarkID')['value'].apply(' '.join).reset_index()
    
    model = Word2Vec(bm['value'])

    model.save('./bookmark_embeddings.bin')


if __name__ == '__main__':
    main()