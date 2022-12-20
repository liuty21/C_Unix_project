CC ?= gcc
BLASLIB = libcblas.a libblas.a
CFLAGS = -g -lstdc++ -lgfortran -lpthread -lX11



CFILES = basic_components.cpp Models.cpp
CFILES_MAIN = $(CFILES) main.cpp
OBJS_MAIN = $(CFILES_MAIN:.cpp=.o)

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

main: $(OBJS_MAIN)
	$(CC) $(CFLAGS) -o $@ $^ $(BLASLIB)

do_Alexnet: main
	./$^ 0

do_Resnet: main
	./$^ 1

do_VGGnet: main
	./$^ 2

clean_main: $(OBJS_MAIN) main
	rm $^