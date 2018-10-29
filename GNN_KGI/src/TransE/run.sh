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
        contain $f "_O"
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
            contain $file "_O"
            if [ $? = 1 ] 
            then 
                echo $1"/"$file"/GI/GI"
                if ! ls $1"/"$file"/GI/GI/"*.bern >/dev/null 2>&1
                    then ./Train_TransE1 $1"/"$file"/GI/GI" &
                fi
                # ./Train_TransE1 $1"/"$file"/subG1" &
                # ./Train_TransE1 $1"/"$file"/subG2" &
                read_dir $1"/"$file
            fi
        fi
    done
}

#测试目录 test
dir=../../data
# travFolder $dir
read_dir $dir
