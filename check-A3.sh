# Script for checking A3.tar.gz file for Assignment 3 subimssion

echo Checking that input_data is available
ls -l input_data/ellipse_N_00010.gal || exit 1

echo Checking that ref_output_data is available
ls -l ref_output_data/ellipse_N_00010_after200steps.gal || exit 1

echo Checking that compare_gal_files program is available
ls -l compare_gal_files/compare_gal_files.c || exit 1

# Remove old tmpdir_for_checking if it exists
rm -rf tmpdir_for_checking

echo Creating temporary directory tmpdir_for_checking

# Create new tmpdir_for_checking directory
mkdir tmpdir_for_checking || exit 1

echo Copying A3.tar.gz file into tmpdir_for_checking directory

cp A3.tar.gz tmpdir_for_checking || exit 1

# cd into the tmpdir_for_checking directory
cd tmpdir_for_checking || exit 1

echo Unpacking the A3.tar.gz file
tar -xzf A3.tar.gz || exit 1

echo Doing cd into A3 directory
cd A3 || exit 1

echo Checking that report.pdf file exists
ls -l report.pdf || exit 1

echo Checking that best_timing.txt file exists
ls -l best_timing.txt || exit 1

echo Checking that no .o files are included
find . -name \*.o | grep o && exit 1

echo Checking that galsim executable is not included
find . -name galsim | grep galsim && exit 1

echo Checking that no .gal files are included
find . -name \*.gal | grep gal && exit 1

echo Checking that make gives the galsim executable
make || exit 1
ls -l galsim || exit 1

echo Checking that doing make again gives \"is up to date\" message
make galsim | grep "is up to date" || exit 1

echo Checking that make clean removes .o files and galsim file
make clean || exit 1
find . -name \*.o | grep o && exit 1
find . -name galsim | grep galsim && exit 1

echo Doing make again
make || exit 1
ls -l galsim || exit 1

echo Checking that running galsim executable works
./galsim 10 ../../input_data/ellipse_N_00010.gal 200 1e-5 0 || exit 1

echo Checking that result.gal file was created
ls -l result.gal || exit 1

echo Compiling compare_gal_files program
gcc -o compare_gal_files ../../compare_gal_files/compare_gal_files.c -lm || exit 1
ls -l compare_gal_files || exit 1

echo Using compare_gal_files program to check result.gal file
./compare_gal_files 10 result.gal ../../ref_output_data/ellipse_N_00010_after200steps.gal > tmp.txt || exit 1
cat tmp.txt
grep pos_maxdiff tmp.txt | grep 00000000 || exit 1

# If we get to this point, then all the different tests above have passed.
echo
echo Congratulations, your A3.tar.gz file seems OK!
echo
