CC = mpicc
CFLAGS =
LIBS = -lm
OBJECTS = MLP.o
EXECUTABLE = MLP

$(EXECUTABLE) : $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $@ $(LIBS)

%.o : %.c
	$(CC) $(CFLAGS) -c $<

.PHONY: clean
clean:
	rm -f *.o $(EXECUTABLE)
