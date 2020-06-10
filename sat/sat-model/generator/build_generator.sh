
SOURCES="problem generator_main satassert"

mkdir -p build_generator

for f in $SOURCES
do
g++ -o "build_generator/${f}.o" -c -O3 -Wall -ggdb -flto -DNDEBUG "src/${f}.cpp"
outputs="${outputs} build_generator/${f}.o"
done

g++ -o build_generator/generator -flto -O3 -Wall -ggdb -DNDEBUG ${outputs}
