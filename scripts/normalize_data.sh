
for ARGUMENT in "$@"
do

    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   

    case "$KEY" in
            INPUT_PATH)              INPUT_PATH=${VALUE} ;;
            OUTPUT_PATH)    OUTPUT_PATH=${VALUE} ;;     
            *)   
    esac    


done


mkdir -p "$OUTPUT_PATH/temp"

python ../mermaid/mermaid_apps/normalize_image_intensities.py --dataset_directory_to_compute_cdf $INPUT_PATH --save_average_cdf_to_file "$OUTPUT_PATH/temp/average_cdf.pt"
echo "Calculated the average CDFs, now normalizing images."

python ../mermaid/mermaid_apps/normalize_image_intensities.py --load_average_cdf_from_file "$OUTPUT_PATH/temp/average_cdf.pt"  --directory_to_normalize $INPUT_PATH --desired_output_directory $OUTPUT_PATH

rm -rf "$OUTPUT_PATH/temp"
rm quantile_averaging.pdf quantile_function.pdf standardized_average_cdf.pdf
echo "Normalized images are generated in $OUTPUT_PATH"
