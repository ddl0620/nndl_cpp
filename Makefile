# Compiler
CXX = g++
CXXFLAGS = -Wall -Wextra -O2 -std=c++17

# Directories
SRC_DIR = .
MATH_LIB_DIR = math_lib
NETWORK_DIR = network
DATA_LOADER_DIR = loader

# Source files
SRCS = $(SRC_DIR)/main.cpp \
       $(MATH_LIB_DIR)/math_lib.cpp \
       $(NETWORK_DIR)/network.cpp \
       $(DATA_LOADER_DIR)/data_loader.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Executable name
TARGET = nndl_cpp

# Default rule
all: $(TARGET)

# Link all object files into the final executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Clean up compiled files
clean:
	rm -f $(OBJS) $(TARGET)

# Run the program
run: all
	./$(TARGET)
