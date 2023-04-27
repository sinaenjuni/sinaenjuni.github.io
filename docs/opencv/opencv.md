---
layout: default
title: OpenCV
nav_order: 3
has_children: true
permalink: /docs/opencv
---
# OpenCV
{: .no_toc }


{:toc}

# Opencv 주요 클래스 정리

## InputArray와 OutputArray
OpenCV에서 입력과 출력에 사용되는 프록시 클래스이다. 다음과 같이 OutputArray는 InputArray를 상속 받아서 만들어진 객체이고, IunputOutputArray는 OutputArray를 상속 받아서 만들어진 객체이다. 때문이 이들 간에는 상속 관계에 의한 호환이 가능하다.
```cpp
class CV_EXPORTS _InputArray {...}
class CV_EXPORTS _OutputArray : public _InputArray {...}
class CV_EXPORTS _InputOutputArray : public _OutputArray {...}
```

## Mat 클래스
행렬을 표현하기 위한 용도로 단일 채널 또는 n차원의 채널을 나타낼 수 있다. Mat 클래스는 정수, 실수 또는 복소수 벡터 및 행렬, 회색조 또는 컬러 이미지, 복셀 볼륨, 벡터 필드, 포인트 클라우드, 텐서, 히스토그램을 저장하는데 사용된다.

### Mat class의 자료형

| 재정의 이름 | 의미	| Depth_mask | 범위 |
|:------|:---------------------------------------------------|:-|:---------------------------|
|CV_8U  |    unsigned char   8-bit unsigned integers 	     |0|	0 ~ 255|
|CV_8S  |    signed char     8-bit signed integers           |1|	-128 ~ 127|
|CV_16U |    unsigned short  16-bit unsigned integers	     |2|	0 ~ 65535|
|CV_16S |    signed short    16-bit signed integers	         |3|	-32768 ~ 32767|
|CV_32S |    int             32-bit signed integers	         |4|	-2147483648| ~ 2147483647|
|CV_32F |    float           32-bit floating-point numbers   |5|	-FLT_MAX ~ FLT_MAX, INF, NAN|
|CV_64F |    double          64-bit floating-point numbers   |6|	-DBL_MAX ~ DBL_MAX, INF, NAN|
|CV_16F |    float16         16-bit floating-point numbers   |7|	-OpenCV 4부터 지원 / 정보 없음|

### Mat 객체 생성 방법
Mat class는 다양한 생성자와 초기화 방법이 구현되어 있다. 가장 먼저 빈 객체를 생성하는 방법이다. 카메라로 부터 받을 데이터를 저장하는 용도로 사용될 수 있다.
```cpp
Mat img1;
// empty matrix
```
다음 방법은 행렬의 크기와 저장할 데이터의 타입을 지정하여 객체를 생성하는 방법이다. 정수 타입의 인자로 Mat 객체를 생성하는 경우 height, width 순서대로 넣어주어야 한다.
```cpp
Mat img2(480, 640, CV_8UC1);
// unsigned char, 1-channel
Mat img3(480, 640, CV_8UC3);		
// unsigned char, 3-channels
```
정수 형태의 인자 대신 Size() 객체를 이용해서 초기화가 가능하다. Size()는 영상 또는 사각형의 크기를 정의위해 사용되는 템플릿 클래스이다. 이때 위 방법과 달리 width, height 순으로 입력할 수 있다.
```cpp
Mat img4(Size(640, 480), CV_8UC3);	
// Size(width, height)
```

### Mat 객체 초기화 방법
먼저 생성과 동시에 데이터를 초기화 하는 방법이다. Scalar()는 크기 4의 double 타입의 배열을 가지고 있는 클래스이다. 각각 영상의 색을 나타내는 B G R A을 저장하며, 배열과 마찬가지로 []연산자를 통해 데이터에 접근이 가능하다.
```cpp
Mat img5(480, 640, CV_8UC1, Scalar(128));		
// initial values, 128
Mat img6(480, 640, CV_8UC3, Scalar(0, 0, 255));	
// initial values, red
Mat mat5 = (Mat_<float>(2, 3) << 1, 2, 3, 4, 5, 6);
Mat mat6 = Mat_<uchar>({2, 3}, {1, 2, 3, 4, 5, 6});
```
Mat의 멤버 함수를 이용한 초기화 방법이다. zeros, ones, eye는 각각 행렬 전체를 0이나 1로 혹은 단위 행렬 형태로 초기화 하는 기능을 수행한다.
```cpp
Mat mat1 = Mat::zeros(3, 3, CV_32SC1);	
// 0's matrix
Mat mat2 = Mat::ones(3, 3, CV_32FC1);	
// 1's matrix
Mat mat3 = Mat::eye(3, 3, CV_32FC1);	
// identity matrix
```
이미 생성된 객체에 데이터를 추가하거나, 다른 크기, 데이터 타입의 형태로 변수를 초기화하는 방법이다.
```cpp
mat4.create(256, 256, CV_8UC3);	// uchar, 3-channels
mat5.create(4, 4, CV_32FC1);	// float, 1-channel
mat4 = Scalar(255, 0, 0);
mat5.setTo(1.f);
```
마지막으로 1차원 배열을 통해 데이터를 초기화 할 수 있다. 다만 행렬 전체 크기와 배열의 크기가 일치해야 한다.
```cpp
float data[] = {1, 2, 3, 4, 5, 6};
Mat mat4(2, 3, CV_32FC1, data);
```

### Mat 객체의 유용한 멤버 함수들
```cpp
cout << frame.depth() << endl;
// 행렬의 원소의 자료형을 반환하는 매크로 상수이다.
// 8:1byte, U: unsigned, S: signed
// #define CV_8U   0
// #define CV_8S   1
// #define CV_16U  2
// #define CV_16S  3
// #define CV_32S  4
// #define CV_32F  5
// #define CV_64F  6
// #define CV_16F  7

cout << frame.channels() << endl;
// 이미지의 채널 수를 출력하는 매크로 상수이다.
// 흑백의 경우 1채널, 컬러 영상의 경우 3채널이다.

cout << frame.type() << endl;
// 행렬의 깊이와 채널을 한 번에 출력하는 매크로 상수이다.
```

## Vec 클래스
벡터(vector)는 같은 동일한 타입의 데이터를 여러개 묶어놓은 형태로 저장하는 클래스이다. `[]`에 대한 연산자 오버로딩을 지원하기 때문에 행렬과 동일하게 데이터 원소 접근이 가능하다. 또한 `<<` 연사자를 오버로딩 하고 있기 때문에 std::cout을 통해 출력이 가능하다.
```cpp
template<typename _Tp, int cn> class Vec : public Matx<_Tp, cn, 1>
{
public:
    ...
    //! default constructor
    Vec();

    Vec(_Tp v0); //!< 1-element vector constructor
    Vec(_Tp v0, _Tp v1); //!< 2-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2); //!< 3-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3); //!< 4-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4); //!< 5-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5); //!< 6-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6); //!< 7-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7); //!< 8-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8); //!< 9-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8, _Tp v9); //!< 10-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8, _Tp v9, _Tp v10, _Tp v11, _Tp v12, _Tp v13); //!< 14-element vector constructor
    explicit Vec(const _Tp* values);

    //! per-element multiplication
    Vec mul(const Vec<_Tp, cn>& v) const;

    //! conjugation (makes sense for complex numbers and quaternions)
    Vec conj() const;

    /*!
      cross product of the two 3D vectors.

      For other dimensionalities the exception is raised
    */
    Vec cross(const Vec& v) const;
    //! conversion to another data type
    template<typename T2> operator Vec<T2, cn>() const;

    /*! element access */
    const _Tp& operator [](int i) const;
    _Tp& operator[](int i);
    const _Tp& operator ()(int i) const;
    _Tp& operator ()(int i);
    ...
};
```

자주 사용되는 `Vec` 클래스의 경우 Vec2(데이터의 수)b(데이터 타입) 으로 사용할 수 있도록 정의되어 있다.
```cpp
typedef Vec<uchar, 2> Vec2b;
typedef Vec<uchar, 3> Vec3b;
typedef Vec<uchar, 4> Vec4b;

typedef Vec<short, 2> Vec2s;
typedef Vec<short, 3> Vec3s;
typedef Vec<short, 4> Vec4s;

typedef Vec<ushort, 2> Vec2w;
typedef Vec<ushort, 3> Vec3w;
typedef Vec<ushort, 4> Vec4w;

typedef Vec<int, 2> Vec2i;
typedef Vec<int, 3> Vec3i;
typedef Vec<int, 4> Vec4i;
typedef Vec<int, 6> Vec6i;
typedef Vec<int, 8> Vec8i;

typedef Vec<float, 2> Vec2f;
typedef Vec<float, 3> Vec3f;
typedef Vec<float, 4> Vec4f;
typedef Vec<float, 6> Vec6f;

typedef Vec<double, 2> Vec2d;
typedef Vec<double, 3> Vec3d;
typedef Vec<double, 4> Vec4d;
typedef Vec<double, 6> Vec6d;
```

## Scalar 클래스
크기가 4인 double 타입의 배열을 멤버 변수로 가지고 있는 클래스이다. 4채널 이하의 영상 데이터의 한 픽셀을 표현하는데 사용된다. `[]` 연사자를 오버로딩 하고있어 행렬과 동일한 방법으로 데이터 원소 접근이 가능하다.
```cpp
template<typename _Tp> class Scalar_ : public Vec<_Tp, 4>
{
public:
    //! default constructor
    Scalar_();
    Scalar_(_Tp v0, _Tp v1, _Tp v2=0, _Tp v3=0);
    Scalar_(_Tp v0);

    Scalar_(const Scalar_& s);
    Scalar_(Scalar_&& s) CV_NOEXCEPT;

    Scalar_& operator=(const Scalar_& s);
    Scalar_& operator=(Scalar_&& s) CV_NOEXCEPT;

    template<typename _Tp2, int cn>
    Scalar_(const Vec<_Tp2, cn>& v);

    //! returns a scalar with all elements set to v0
    static Scalar_<_Tp> all(_Tp v0);

    //! conversion to another data type
    template<typename T2> operator Scalar_<T2>() const;

    //! per-element product
    Scalar_<_Tp> mul(const Scalar_<_Tp>& a, double scale=1 ) const;

    //! returns (v0, -v1, -v2, -v3)
    Scalar_<_Tp> conj() const;

    //! returns true iff v1 == v2 == v3 == 0
    bool isReal() const;
};
```
기본적으로 double 데이터 타입으로 정의되어 있다.
```cpp
typedef Scalar_<double> Scalar;
```

## 행렬 연산 관련된 함수
{: .note } `mask` 인자의 경우 0 이 아닌 위치의 데이터 원소들만 연산에 사용한다.
### sum
행렬의 합을 구하는 함수이다. Scalar 데이터 타입의 각 채널에 대한 행렬의 합을 반환한다.

```cpp
Scalar sum(InputArray src);
```

### mean
평균을 구하는 함수이다. sum 함수와 마찬가지로 각 채널에 대한 평균을 Scalar 형태의 데이터 타입을 반환한다.
```cpp
Scalar mean(InputArray src, InputArray mask = noArray());
```


### minMaxLoc
```cpp
void minMaxLoc(InputArray src, CV_OUT double* minVal,
                CV_OUT double* maxVal = 0, CV_OUT Point* minLoc = 0,
                CV_OUT Point* maxLoc = 0, InputArray mask = noArray());
````

### converTo

```cpp
void GpuMat::convertTo(OutputArray dst, int rtype, double alpha, double beta) const
{
    convertTo(dst, rtype, alpha, beta, Stream::Null());
}
```


### normalize


```cpp
void normalize( InputArray src, InputOutputArray dst, double alpha = 1, double beta = 0,
                int norm_type = NORM_L2, int dtype = -1, InputArray mask = noArray());
```


```cpp
enum NormTypes {
                NORM_INF       = 1,
                NORM_L1        = 2,
                NORM_L2        = 4,
                NORM_L2SQR     = 5,
                NORM_HAMMING   = 6,
                NORM_HAMMING2  = 7,
                NORM_TYPE_MASK = 7, //!< bit-mask which can be used to separate norm type from norm flags
                NORM_RELATIVE  = 8, //!< flag
                NORM_MINMAX    = 32 //!< flag
               };
```


### Exampl code

```cpp
void ex_ops(){
    uchar data[] = {1,2,3,4,5,6};
    Mat mat(2,3, CV_8UC1, data);
    cout << mat << endl;

    int ret_sum = (int)sum(mat)[0];
    cout << ret_sum << endl;

    double ret_mean = (double)mean(mat)[0];
    cout << ret_mean << endl;

    double minval, maxval;
    Point minloc, maxloc;
    minMaxLoc(mat, &minval, &maxval, &minloc, &maxloc);
    cout << format("%f %f (%d, %d) (%d, %d)", 
    minval, maxval, 
    minloc.x, minloc.y, 
    maxloc.x, maxloc.y) << endl;
    
    Mat cvt_mat;
    mat.convertTo(cvt_mat, CV_32FC1, 0.1, -5);
    cout << cvt_mat << endl;
    
    uchar maks_data[] = {1,1,1,1,1,0};
    Mat mask(2,3, CV_8UC1, maks_data);
    cout << mask << endl;

    Mat norm_mat;
    normalize(mat, norm_mat, 0, 1, NORM_MINMAX, CV_32FC1, mask);
    cout << norm_mat << endl;

}
```


```text
[  1,   2,   3;
   4,   5,   6]
21
3.5
1.000000 6.000000 (0, 0) (2, 1)
[-4.9000001, -4.8000002, -4.6999998;
 -4.5999999, -4.5, -4.4000001]
[  1,   1,   1;
   1,   1,   0]
[0, 0.25, 0.5;
 0.75, 1, 0]
```



## Point_ 클래스
2차원 점의 좌표를 표현하기 위한 템플릿 클래스이다. 좌표를 나타내는 x와 y 두 멤버 변수를 가지고 있다. 또한 내적을 구하는 `dot()` 함수, double 타입의 내적을 구하는 `ddot()` 함수, 외적을 구하는 `cross()`, 사격형의 겹치는 부분이 있는지를 확인하는 `inside()`로 4개의 멤버 함수로 구성된다.
```cpp
template<typename _Tp> class Point_
{
public:
    Point_();
    Point_(_Tp _x, _Tp _y);
    ...
    _Tp dot(const Point_& pt) const;
    //! dot product computed in double-precision arithmetics
    double ddot(const Point_& pt) const;
    //! cross-product
    double cross(const Point_& pt) const;
    //! checks whether the point is inside the specified rectangle
    bool inside(const Rect_<_Tp>& r) const;
    _Tp x; //!< x coordinate of the point
    _Tp y; //!< y coordinate of the point
};
```

자주 사용되는 데이터 타입에 대해서 아래와 같이 정의되어 있어 이를 사용해도 된다.
```cpp
typedef Point_<int> Point2i;
typedef Point_<int64> Point2l;
typedef Point_<float> Point2f;
typedef Point_<double> Point2d;
typedef Point2i Point;
```

다양한 사칙 연산에 대한 연산자 오버로딩과 std::cout 출력을 위한 << 연산자 오버로딩을 지원한다.
```cpp
pt1 = pt2 + pt3;
pt1 = pt2 - pt3;
pt1 = pt2 * a;
pt1 = a * pt2;
pt1 = pt2 / a;
pt1 += pt2;
pt1 -= pt2;
pt1 *= a;
pt1 /= a;
double value = norm(pt); // L2 norm
pt1 == pt2;
pt1 != pt2;
```

## Size_ 클래스
영상 또는 사각형의 크기를 표현하기 위한 템플릿 클래스이다. `width`, `height` 두 멤버 변수를 가지고 있으며, 면적을 구하는 `area()` 함수가 구현되어 있다.
```cpp
template<typename _Tp> class Size_
{
public:
    typedef _Tp value_type;

    //! default constructor
    Size_();
    Size_(_Tp _width, _Tp _height);
    ...
    _Tp area() const;
    //! aspect ratio (width/height)
    double aspectRatio() const;
    //! true if empty
    bool empty() const;

    //! conversion of another data type.
    template<typename _Tp2> operator Size_<_Tp2>() const;

    _Tp width; //!< the width
    _Tp height; //!< the height
};
```
`point_` 클래스와 마찬가지로 사칙 연사에 대한 오버로딩과 << 연산자에 대한 오버로딩을 지원한다.

또한 자주 사용되는 데이터 타입에 대해서 다음과 같이 정의되어 있다.
```cpp
typedef Size_<int> Size2i;
typedef Size_<int64> Size2l;
typedef Size_<float> Size2f;
typedef Size_<double> Size2d;
typedef Size2i Size;
```


## Rect_ 클래스
2차원의 사각형 표현을 위함 템플릿 클래스이다. 좌상단과 넓이와 높이를 가지고 정의되며, `x`, `y`, `width`, `height` 멤버 변수를 가지고 있다. 또한 사각형의 좌상단, 우하단을 반환하는 `tl()`, `br()`, 점이 사각형 안에 존재하는지를 확인해 주는 `contains()` 멤버 함수가 정의되어 있다. 
```cpp
template<typename _Tp> class Rect_
{
public:
    typedef _Tp value_type;

    //! default constructor
    Rect_();
    Rect_(_Tp _x, _Tp _y, _Tp _width, _Tp _height);
    ...
    //! the top-left corner
    Point_<_Tp> tl() const;
    //! the bottom-right corner
    Point_<_Tp> br() const;

    //! size (width, height) of the rectangle
    Size_<_Tp> size() const;
    //! area (width*height) of the rectangle
    _Tp area() const;
    //! true if empty
    bool empty() const;

    //! conversion to another data type
    template<typename _Tp2> operator Rect_<_Tp2>() const;

    //! checks whether the rectangle contains the point
    bool contains(const Point_<_Tp>& pt) const;

    _Tp x; //!< x coordinate of the top-left corner
    _Tp y; //!< y coordinate of the top-left corner
    _Tp width; //!< width of the rectangle
    _Tp height; //!< height of the rectangle
};
```
자주 사용되는 데이터 타입에 대해서 다음과 같이 정의되어 있다.
```cpp
typedef Rect_<int> Rect2i;
typedef Rect_<float> Rect2f;
typedef Rect_<double> Rect2d;
typedef Rect2i Rect;
```
Rect 객체와 Size, Point 객체의 덧셈과 뺼셈, Rect 객체 간의 논리 연산을 지원한다.
```cpp
Rect rc1; // [0 x 0 from (0, 0)]
Rect rc2(10, 10 60, 40); // [60 x 40 from (10, 10)]
Rect rc3 = rc1 + Size(50, 40); // [50 x 40 from (0, 0)]
Rect rc4 = rc2 + Point(10, 10); // [60 x 40 from (20, 20)]
Rect rc5 = rc3 & rc4; // [30 x 20 from (20, 20)]
Rect rc6 = rc3 | rc4; // [80 x 60 from (0, 0)]
```

## Range 클래스
정수 데이터 타입의 범위를 나타내는 클래스이다. 시작과 끝을 나타내는 두 멤버 변수를 가지고 있으며, 이때 시작은 범위에 포함되고 끝은 포함되지 않는다`(e.g.(시작:끝])`.
```cpp
class CV_EXPORTS Range
{
public:
    Range();
    Range(int _start, int _end);
    int size() const;
    bool empty() const;
    static Range all();

    int start, end;
};
```


# Opencv 주요 함수 정리
## 창 관련 함수
### namedWindow (새로운 창을 생성)
```cpp
void namedWindow(const String& winname, int flags = WINDOW_AUTOSIZE);
```
```cpp
//! Flags for cv::namedWindow
enum WindowFlags {
       WINDOW_NORMAL     = 0x00000000, 
       //!< the user can resize the window (no constraint) / also use to switch a fullscreen window to a normal size.
       WINDOW_AUTOSIZE   = 0x00000001, 
       //!< the user cannot resize the window, the size is constrainted by the image displayed.
       WINDOW_OPENGL     = 0x00001000, 
       //!< window with opengl support.
       WINDOW_FULLSCREEN = 1,          
       //!< change the window to fullscreen.
       WINDOW_FREERATIO  = 0x00000100, 
       //!< the image expends as much as it can (no ratio constraint).
       WINDOW_KEEPRATIO  = 0x00000000, 
       //!< the ratio of the image is respected.
       WINDOW_GUI_EXPANDED=0x00000000, 
       //!< status bar and tool bar
       WINDOW_GUI_NORMAL = 0x00000010, 
       //!< old fashious way
    };
```

### moveWindow (창의 위치를 이동)
주어진 이름의 창의 위치를 x, y 좌료로 이동시키는 함수이다.
```cpp
void moveWindow(const String& winname, int x, int y);
```


### resizeWindow (창의 크기를 조정)
주어진 이름의 창의 크기를 조정하는 함수이다. 창 생성시 `WINDOW_NORMAL` 속성으로 생성되어야 하며, `imshow()` 함수보다 나중에 호출 되어야 정상적으로 동작한다.
```cpp
void resizeWindow(const String& winname, int width, int height);


```

### waitKey (키 입력 대기)
키 입력을 대기하는 함수로 프로그램이 종료되어 창이 닫히는 것을 방지한다. 인자로 정수 형태로 시간을 입력받는다. 이때 정수는 milliseconds 단위이다. 또한 입력받을 키의 코드`(ESC=27, ENTER=13, TAB=9)`를 반환한다. 특수 키를 입력받고 싶다면 `waitKetEx()`함수를 이용하면 된다. 스페이스바의 경우 공백(' ')과 비교하면 된다.

```cpp
int waitKey(int delay = 0);
int waitKeyEx(int delay = 0);
```


### 창 종료
주어진 창의 이름을 닫는 함수이다.
```cpp
void destroyWindow(const String& winname);
```
열려있는 모든 창을 닫는 함수이다.
```cpp
void destroyAllWindows();
```

---
## 영상 관련 함수
### imread (영상 파일 읽기)
이미지 파일을 Mat 객체로 반환하는 함수이다. flags는 이미지 파일을 불러올 때, 색상 옵션을 설정하는 부분이다. 주로 `IMREAD_UNCHANGED`, `IMREAD_GRAYSCALE` 그리고 `IMREAD_COLOR`를 주로 사용한다. 각 옵션은 원본 이미지의 색상 특성을 그대로 반영해서 불러오는 방법이며, 흑백 그리고 색상 모드로 이미지를 불러오는 방법이다.
```cpp
Mat imread( const String& filename, int flags = IMREAD_COLOR );
```
```cpp
//! Imread flags
enum ImreadModes {
       IMREAD_UNCHANGED            = -1, 
       //!< If set, return the loaded image as is (with alpha channel, otherwise it gets cropped). Ignore EXIF orientation.
       IMREAD_GRAYSCALE            = 0,  
       //!< If set, always convert image to the single channel grayscale image (codec internal conversion).
       IMREAD_COLOR                = 1,  
       //!< If set, always convert image to the 3 channel BGR color image.
       IMREAD_ANYDEPTH             = 2,  
       //!< If set, return 16-bit/32-bit image when the input has the corresponding depth, otherwise convert it to 8-bit.
       IMREAD_ANYCOLOR             = 4,  
       //!< If set, the image is read in any possible color format.
       IMREAD_LOAD_GDAL            = 8,  
       //!< If set, use the gdal driver for loading the image.
       IMREAD_REDUCED_GRAYSCALE_2  = 16, 
       //!< If set, always convert image to the single channel grayscale image and the image size reduced 1/2.
       IMREAD_REDUCED_COLOR_2      = 17, 
       //!< If set, always convert image to the 3 channel BGR color image and the image size reduced 1/2.
       IMREAD_REDUCED_GRAYSCALE_4  = 32, 
       //!< If set, always convert image to the single channel grayscale image and the image size reduced 1/4.
       IMREAD_REDUCED_COLOR_4      = 33, 
       //!< If set, always convert image to the 3 channel BGR color image and the image size reduced 1/4.
       IMREAD_REDUCED_GRAYSCALE_8  = 64, 
       //!< If set, always convert image to the single channel grayscale image and the image size reduced 1/8.
       IMREAD_REDUCED_COLOR_8      = 65, 
       //!< If set, always convert image to the 3 channel BGR color image and the image size reduced 1/8.
       IMREAD_IGNORE_ORIENTATION   = 128 
       //!< If set, do not rotate the image according to EXIF's orientation flag.
     };
```

### 영상 출력
주어진 이름의 창에 영상을 출력한다. 만약 창이 없다면 새로운 창을 만들고 그 위에 이미지를 출력한다. 입력 이미지의 데이터 타입이 `InputArray`인 이유는 `Mat` 형식 외에도 다양한 타입의 데이터를 시각화할 수 있도록 하기 위함이다.
```cpp
void imshow(const String& winname, InputArray mat);
```

### (imwrite) 영상 파일 저장
`InputArray` 타입의 변수를 파일로 저장하는 함수이다. 마지막 인자인 `params`는 JPG로 저장할 경우, 압축율을 지정할 때 사용되는 인자이다(`e.g. {IMWRITE_JEPG_QUARITY, 90} : 압축율을 90%로 지정`). 파일 저장이 정상적으로 이루어지면, true 그렇지 않으면 false를 반환한다.
```cpp
bool imwrite( const String& filename, InputArray img, const std::vector<int>& params = std::vector<int>());
```

```cpp
//! Imwrite flags
enum ImwriteFlags {
       IMWRITE_JPEG_QUALITY        = 1,  
       //!< For JPEG, it can be a quality from 0 to 100 (the higher is the better). Default value is 95.
       IMWRITE_JPEG_PROGRESSIVE    = 2,  
       //!< Enable JPEG features, 0 or 1, default is False.
       IMWRITE_JPEG_OPTIMIZE       = 3,  
       //!< Enable JPEG features, 0 or 1, default is False.
       IMWRITE_JPEG_RST_INTERVAL   = 4,  
       //!< JPEG restart interval, 0 - 65535, default is 0 - no restart.
       IMWRITE_JPEG_LUMA_QUALITY   = 5,  
       //!< Separate luma quality level, 0 - 100, default is -1 - don't use.
       IMWRITE_JPEG_CHROMA_QUALITY = 6,  
       //!< Separate chroma quality level, 0 - 100, default is -1 - don't use.
       IMWRITE_JPEG_SAMPLING_FACTOR = 7, 
       //!< For JPEG, set sampling factor. See cv::ImwriteJPEGSamplingFactorParams.
       IMWRITE_PNG_COMPRESSION     = 16, 
       //!< For PNG, it can be the compression level from 0 to 9. A higher value means a smaller size and longer compression time. If specified, strategy is changed to IMWRITE_PNG_STRATEGY_DEFAULT (Z_DEFAULT_STRATEGY). Default value is 1 (best speed setting).
       IMWRITE_PNG_STRATEGY        = 17, 
       //!< One of cv::ImwritePNGFlags, default is IMWRITE_PNG_STRATEGY_RLE.
       IMWRITE_PNG_BILEVEL         = 18, 
       //!< Binary level PNG, 0 or 1, default is 0.
       IMWRITE_PXM_BINARY          = 32, 
       //!< For PPM, PGM, or PBM, it can be a binary format flag, 0 or 1. Default value is 1.
       IMWRITE_EXR_TYPE            = (3 << 4) + 0, /* 48 */ 
       //!< override EXR storage type (FLOAT (FP32) is default)
       IMWRITE_EXR_COMPRESSION     = (3 << 4) + 1, /* 49 */ 
       //!< override EXR compression type (ZIP_COMPRESSION = 3 is default)
       IMWRITE_EXR_DWA_COMPRESSION_LEVEL = (3 << 4) + 2, /* 50 */ 
       //!< override EXR DWA compression level (45 is default)
       IMWRITE_WEBP_QUALITY        = 64, 
       //!< For WEBP, it can be a quality from 1 to 100 (the higher is the better). By default (without any parameter) and for quality above 100 the lossless compression is used.
       IMWRITE_HDR_COMPRESSION     = (5 << 4) + 0, /* 80 */ 
       //!< specify HDR compression
       IMWRITE_PAM_TUPLETYPE       = 128,
       //!< For PAM, sets the TUPLETYPE field to the corresponding string value that is defined for the format
       IMWRITE_TIFF_RESUNIT        = 256,
       //!< For TIFF, use to specify which DPI resolution unit to set; see libtiff documentation for valid values
       IMWRITE_TIFF_XDPI           = 257,
       //!< For TIFF, use to specify the X direction DPI
       IMWRITE_TIFF_YDPI           = 258,
       //!< For TIFF, use to specify the Y direction DPI
       IMWRITE_TIFF_COMPRESSION    = 259,
       //!< For TIFF, use to specify the image compression scheme. See libtiff for integer constants corresponding to compression formats. Note, for images whose depth is CV_32F, only libtiff's SGILOG compression scheme is used. For other supported depths, the compression scheme can be specified by this flag; LZW compression is the default.
       IMWRITE_JPEG2000_COMPRESSION_X1000 = 272 
       //!< For JPEG2000, use to specify the target compression rate (multiplied by 1000). The value can be from 0 to 1000. Default is 1000.
     };
```

---


## 예외처리 관련 함수
Assert는 인자로 주어진 조건이 `true`가 아닌 경우에 프로그램을 종료하는 매크로 함수이다.
```cpp
#define CV_Assert( expr ) do { if(!!(expr)) ; else cv::error( cv::Error::StsAssert, #expr, CV_Func, __FILE__, __LINE__ ); } while(0)
```