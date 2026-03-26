CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -std=c99
LDFLAGS = -lm

SRC     = src/main.c src/data.c src/nn.c
OBJ     = $(SRC:.c=.o)
TARGET  = diabetes_nn

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $@ $(LDFLAGS)

src/%.o: src/%.c
	$(CC) $(CFLAGS) -Isrc -c $< -o $@

run: all
	./$(TARGET)

clean:
	rm -f src/*.o $(TARGET)
