# strA="helloworld"
# strB="low"
# contain $strA $strB
# echo $?
function contain(){
    if [[ $1 =~ $2 ]]
    then
        return 1
    else
        return 0
    fi
}
# 2 in 1
function travFolder(){ 
    flist=`tree `
    for f in $flist
    do
        contain $f "Ove"
        if [ $? = 0 ]
        then
            echo "$f"
        fi
    done
    cd ../ 
}

function read_dir(){
    for file in `ls $1`
    do
        if [ -d $1"/"$file ]
        then
            contain $file "_subGraph1"
            if [ $? = 1 ] 
            then 
                echo $1"/"$file
                echo $1"/"$file"/subG1"
                ./Train_TransE $1"/"$file"/subG1" &
                ./Train_TransE $1"/"$file"/subG2" &
            else
                read_dir $1"/"$file
            fi
        fi
    done
}

#测试目录 test
dir=/home/dreamer/codes/my_code/KGI/sampling/results/fb15k
# dir=/home/dreamer/codes/my_code/KGI/sampling/results/fb15k/50_70_4/uniform

# read_dir $dir
./Train_TransE /home/dreamer/codes/my_code/KGI/sampling/results/fb15k/50_70_4/uniform/subG1 &
./Train_TransE /home/dreamer/codes/my_code/KGI/sampling/results/fb15k/50_70_4/uniform/subG2 &
