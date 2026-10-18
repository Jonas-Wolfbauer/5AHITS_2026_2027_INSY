# Forward-only inference in C (parts 4 and 5). Needs gcc and make, e.g. w64devkit.
#
#   make         build all programs into build/
#   make test    build, then run the self-test and both programs
#   make MODELS=../models/csharp test    the same with the networks trained in C#

CC = gcc
CFLAGS = -std=c17 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
LDLIBS = -lm
MODELS ?= ../models/python
BUILD = build

ifeq ($(OS),Windows_NT)
    EXE = .exe
endif

# The generated headers include neural_network.h from this folder
HEADER_FLAGS = -I. -I$(MODELS)

PROGRAMS = $(BUILD)/predict_txt$(EXE) $(BUILD)/predict_header$(EXE) $(BUILD)/selftest$(EXE)

all: $(PROGRAMS)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/predict_txt$(EXE): predict_txt.c evaluation.c neural_network.c evaluation.h neural_network.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ predict_txt.c evaluation.c neural_network.c $(LDLIBS)

# Rebuilt every time: the headers depend on MODELS
$(BUILD)/predict_header$(EXE): FORCE | $(BUILD)
	$(CC) $(CFLAGS) $(HEADER_FLAGS) -o $@ predict_header.c evaluation.c neural_network.c $(LDLIBS)

$(BUILD)/selftest$(EXE): FORCE | $(BUILD)
	$(CC) $(CFLAGS) $(HEADER_FLAGS) -o $@ selftest.c neural_network.c $(LDLIBS)

test: all
	$(BUILD)/selftest$(EXE) $(MODELS)
	$(BUILD)/predict_txt$(EXE) $(MODELS)/xor.txt
	$(BUILD)/predict_txt$(EXE) $(MODELS)/seven_segment.txt
	$(BUILD)/predict_header$(EXE)

clean:
	rm -rf $(BUILD)

FORCE:

.PHONY: all test clean FORCE
