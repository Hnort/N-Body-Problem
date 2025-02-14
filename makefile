CC = gcc
CFLAGS = -Wall -O3 -fopt-info-vec -lm -lX11
OBJS = galsim.o graphics.o

all: galsim

galsim: $(OBJS)
	$(CC) -o galsim $(OBJS) $(CFLAGS)

galsim.o: galsim.c graphics.h
	$(CC) -c galsim.c $(CFLAGS)

graphics.o: graphics.c graphics.h
	$(CC) -c graphics.c $(CFLAGS)

clean:
	rm -f galsim $(OBJS)
