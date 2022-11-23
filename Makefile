CC ?= gcc
BLASLIB = libcblas.a libblas.a
CFLAGS = -g -lstdc++ -lgfortran
BSUB = bsub -Is


CFILES = basic_components.cpp test_code.cpp
OBJS = $(CFILES:.cpp=.o)

foo: $(OBJS)
	$(BSUB) $(CC) $(CFLAGS) -o $@ $^ $(BLASLIB)

%.o: %.cpp
	$(BSUB) $(CC) $(CFLAGS) -c -o $@ $<

test: foo
	$(BSUB) ./$^

test_blas: test_blas.cpp
	$(BSUB) $(CC) $(CFLAGS) -o $@ $^ $(BLASLIB)

do_test_blas: test_blas
	$(BSUB) ./$^

clean: $(OBJS) foo
	rm $^