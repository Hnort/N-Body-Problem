CFLAGS=-Wall
INCLUDES=-I/opt/X11/include
LDFLAGS=-L/opt/X11/lib -lX11 -lm

graphics_test: graphics_test.o graphics.o
	gcc -o graphics_test graphics_test.o graphics.o $(LDFLAGS)

graphics_test.o: graphics_test.c graphics.h
	gcc $(CFLAGS) $(INCLUDES) -c graphics_test.c

graphics.o: graphics.c graphics.h
	gcc $(CFLAGS) $(INCLUDES) -c graphics.c

clean:
	rm -f ./graphics_test *.o
