This project is on developing and evaluation of a systematic knowledge engineering approach for fine-grained annotation of textbooks with underlying knowledge in the form of concepts.

## Data

The dataset includes two parts: all the annotation files and some section samples from [IIR textbook](https://nlp.stanford.edu/IR-book/information-retrieval-book.html).

The input text file follows the format: <section_id>\<tab>\<section title>\<tab>\<section keywords>\<tab>\<section content>
```
iir_1	Boolean retrieval	"INFORMATION RETRIEVAL, GREP, INDEX, INCIDENCE MATRIX, TERM"	"The meaning of the term information retrieval can be very broad. Just getting a credit card out of your wallet so that you can type in the card number is a form of information retrieval..."
iir_1_1	An example information retrieval problem	"AD HOC RETRIEVAL, INFORMATION NEED, QUERY, RELEVANCE"	"A fat book which many people own is Shakespeare's Collected Works. Suppose you wanted to determine which plays of Shakespeare contain the words Brutus AND Caesar and NOT Calpurnia..."
```

## Cite
If you use the dataset please cite the following paper:
> Wang, M., Chau, H., Thaker, K. et al. Knowledge Annotation for Intelligent Textbooks. Tech Know Learn (2021). [[Paper]](https://link.springer.com/article/10.1007%2Fs10758-021-09544-z)
```
@article{wang2021knowledge,
  title={Knowledge Annotation for Intelligent Textbooks},
  author={Wang, Mengdi and Chau, Hung and Thaker, Khushboo and Brusilovsky, Peter and He, Daqing},
  journal={Technology, Knowledge and Learning},
  pages={1--22},
  year={2021},
  publisher={Springer}
}