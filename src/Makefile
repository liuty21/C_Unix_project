CC ?= gcc
BLASLIB = libcblas.a libblas.a
CFLAGS = -g -lstdc++ -lgfortran -lpthread -lX11



CFILES = basic_components.cpp Models.cpp
CFILES_TEST = $(CFILES) test_code.cpp
CFILES_MAIN = $(CFILES) main.cpp
OBJS_TEST = $(CFILES_TEST:.cpp=.o)
OBJS_MAIN = $(CFILES_MAIN:.cpp=.o)

test: $(OBJS_TEST)
	$(CC) $(CFLAGS) -o $@ $^ $(BLASLIB)

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

do_test: test
	./$^

test_blas: test_blas.cpp
	$(CC) $(CFLAGS) -o $@ $^ $(BLASLIB)

do_test_blas: test_blas
	./$^

main: $(OBJS_MAIN)
	$(CC) $(CFLAGS) -o $@ $^ $(BLASLIB)

do_Alexnet: main
	./$^ 0

do_Resnet: main
	./$^ 1

do_VGGnet: main
	./$^ 2

clean_test: $(OBJS_TEST) test
	rm $^

clean_main: $(OBJS_MAIN) main
	rm $^