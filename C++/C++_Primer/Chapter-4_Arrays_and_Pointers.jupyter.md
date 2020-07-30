```c++
#include <iostream>
using namespace std;
```

# Arrays and Pointers
## Arrays
Q: Why do we need arrays? (letters=>words => a sentece; numbers => a sequence; sampling)

An array is a compound type that consists of **a type specifier**, **an identifier**, and **a dimension**. The type specifier indicates what type the elements stored in the array will have. The dimension specifies how many elements the array will contain.

### Defining and Initializing Arrays

```c++
int ragnar[7];
```

![](http://static.zybuluo.com/AustinMxnet/279w37wpp3i2z7esjzo09z0s/image.png)

The dimension must be a **constant** expression whose value is greater than or equal to one.

```c++
const int max_files = 20;
string fileTable[max_files + 1];
```

```c++
//int min_files = 5;
//string fileTable[min_files]; //not allowed
```

#### Explicitly Initializing Array Elements
Unless we explicitly supply element initializers, the elements of a local array of built-in type are **uninitialized**. Using these elements for any purpose other than to assign a new value is **undefined**.

```c++
//int ia[3]; // not recommended
int ia[3] = {0, 1, 2};

ia
```

```c++
int ia1[5] = {};  // all elements set to 0
int ia2[5] = {0}; // all elements set to 0

ia1
```

```c++
int ia3[5] = {1}; //[1, 0, 0, 0, 0]
int ia4[5] = {1, 1, 1, 1, 1};

ia3
```

An explicitly initialized array need not specify a dimension value. The compiler will infer the array size from the number of elements listed:

```c++
int ia5[] = {0, 1, 2}; // an array of dimension 3

ia5
```

If the dimension size is specified, the number of elements provided **must not exceed** that size.

```c++
//int ia6[2] = {0, 1, 2} // not allowed
```

#### Character Arrays Are Special
A character array can be initialized with either a list of comma-separated character literals enclosed in braces or a string literal. Note, that the two forms are not equivalent. When we create a character array from a string literal, **the null is also inserted into the array**:

```c++
char ca1[] = {'C', '+', '+'}; // no null
char ca2[] = {'C', '+', '+', '\0'}; // explicit null
char ca3[] = "C++"; //  null terminator added automatically
```

```c++
//cout << ca1 << endl; // memory leak!
cout << ca2 << endl;
cout << ca3 << endl;
```

```c++
cout << sizeof(ca1) << endl;
cout << sizeof(ca2) << endl;
cout << sizeof(ca3) << endl;
```

It is important to remember the **null-terminator** when initializing an array of characters to a literal. For example, the following is
a compile-time error:

```c++
//const char cname[6] = "Daniel"; // error: Daniel is 7 elements
```

### Operations on Arrays
Array elements, like `vector` elements, may be accessed using the subscript operator. Like the elements of a `vector`, the elements of an array are numbered **beginning with 0**. For an array of ten elements, the correct index values are 0 through 9, not 1 through 10.

When we subscript an array, the right type to use for the index is `size_t`. In the following example, a for loop steps through the 10 elements of an array, assigning to each the value of its index:

```c++
const size_t array_size = 10;
int ia7[array_size]; // 10 ints, elements are uninitialized

// loop through array, assigning value of its index to each element
for (size_t ix = 0; ix != array_size; ++ix)
    ia[ix] = ix;
```

```c++
for (size_t ix = 0; ix != array_size; ++ix)
    cout << ia[ix] << ", ";
```

Using a similar loop, we can copy one array into another:

```c++
int ia8[array_size]; // local array, elements uninitialized
// copy elements from ia1 into ia2
for (size_t ix = 0; ix != array_size; ++ix)
    ia8[ix] = ia7[ix];
```

#### Checking Subscript Values
By far, the most common causes of security problems are so-called "buffer overflow" bugs. These bugs occur when a subscript is not checked and reference is made to an element **outside the bounds of an array** or other similar data structure.

```c++
//int a[5];
//a[6] = 2; // not allowed
```
