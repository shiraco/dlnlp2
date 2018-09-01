
# coding: utf-8
import sys
sys.path.append('..')
import os
import numpy as np

def preprocess(text):
    text = text.lower()
    text = text.replace('.',' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = [word_to_id[w] for w in words]
    corpus = np.array(corpus)

    return corpus, word_to_id, id_to_word


def create_co_matrix(corpus, vocab_size, window_size=1):
    '''共起行列の作成
    :param corpus: コーパス（単語IDのリスト）
    :param vocab_size:語彙数
    :param window_size:ウィンドウサイズ（ウィンドウサイズが1のときは、単語の左右1単語がコンテキスト）
    :return: 共起行列
    '''

    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                rig_word_id = corpus[right_idx]
                co_matrix[word_id, rig_word_id] += 1

    return co_matrix


def cos_similarity(x, y, eps=1e-8):
    '''コサイン類似度の算出
    :param x: ベクトル
    :param y: ベクトル
    :param eps: ”0割り”防止のための微小値
    :return:
    '''

    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)

    return np.dot(nx, ny)


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    '''類似単語の検索
    :param query: クエリ（テキスト）
    :param word_to_id: 単語から単語IDへのディクショナリ
    :param id_to_word: 単語IDから単語へのディクショナリ
    :param word_matrix: 単語ベクトルをまとめた行列。各行に対応する単語のベクトルが格納されていることを想定する
    :param top: 上位何位まで表示するか
    '''

    # クエリの単語ベクトルを取り出す
    if query not in word_to_id:
        print('{} is not found'.format(query))
        return

    print('\n[query] {}'.format(query))
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # コサイン類似度を求める
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # 高い順に出力
    count = 0
    for i in (-1 * similarity).argsort():  # argsort: numpyの要素を小さい順にソート
        if id_to_word[i] == query:
            continue
        print(' {}: {}'.format(id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return


def ppmi(C, verbose=False, eps=1e-8):
    '''PPMI（正の相互情報量）の作成
    :param C: 共起行列
    :param verbose: 進行状況を出力するかどうか
    :return:
    '''

    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[1]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j] * S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total//100) == 0:
                    print('{:1f}%% done'.format(100 * cnt / total))

    return M


if __name__ == '__main__':
    text ='You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)

    print("## corpus")
    print(corpus)

    co_matrix = create_co_matrix(corpus, len(word_to_id))

    print("## co_matrix")
    print(co_matrix)

    print("## most_similar")
    most_similar('you', word_to_id, id_to_word, co_matrix)
