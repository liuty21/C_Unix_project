CC ?= gcc
BLASLIB = libcblas.a libblas.a
CFLAGS = -g -lstdc++ -lgfortran -lpthread -lX11
BSUB = bsub -Is -q gpu


CFILES = basic_components.cpp Models.cpp
CFILES_TEST = $(CFILES) test_code.cpp
CFILES_MAIN = $(CFILES) main.cpp
OBJS_TEST = $(CFILES_TEST:.cpp=.o)
OBJS_MAIN = $(CFILES_MAIN:.cpp=.o)

test: $(OBJS_TEST)
	$(BSUB) $(CC) $(CFLAGS) -o $@ $^ $(BLASLIB)

%.o: %.cpp
	$(BSUB) $(CC) $(CFLAGS) -c -o $@ $<

do_test: test
	$(BSUB) ./$^

test_blas: test_blas.cpp
	$(BSUB) $(CC) $(CFLAGS) -o $@ $^ $(BLASLIB)

do_test_blas: test_blas
	$(BSUB) ./$^

main: $(OBJS_MAIN)
	$(BSUB) $(CC) $(CFLAGS) -o $@ $^ $(BLASLIB)

do_Alexnet: main
	$(BSUB) ./$^ 0

do_Resnet: main
	$(BSUB) ./$^ 1

do_VGGnet: main
	$(BSUB) ./$^ 2

clean_test: $(OBJS_TEST) test
	rm $^

clean_main: $(OBJS_MAIN) main
	rm $^