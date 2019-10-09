# Decision-Tree-from-scratch

In this project, I will implement the Decision Tree learning algorithm from scratch(Only use numpy). 


I will use a data-set that includes mushroom records drawn from the Audubon Society Field Guide to North American
Mushrooms (1981). The database describes samples from different species of gilled mushrooms in
the families Agaricus and Lepiota. Each sample is described by a string of 23 characters, on a single
line of the provided file( mushroom_data.txt); each such string describes the values of 22 attributes
for each sample (described below), and the last character corresponds to the correct classification
of the mushroom into either edible (e) or poisonous (p) mushrooms. 

For example, the first two samples from the data-set are a poisonous and then an edible species, as follows:

x s n t p f c n k e e s s w w p w o p k s u p

x s y t a f c b k e c s s w w p w o p n n g e

The 22 attribute variables, and their values, are given in Table 1, at the end of this document.
(and are also tabulated in the file (properties.txt), for reference).

When the program begins, it should ask the user for three inputs:

A training set size: this should be some integer value S that is a multiple of 250, within
bounds 250 < S < 1000.

A training set increment: this should be some integer value I = {10; 25; 50} (that is,
one of those three values only).

A heuristic for use in choice of attributes when building a tree. This will give the
user a choice between the basic counting heuristic  and the
information-theoretic
