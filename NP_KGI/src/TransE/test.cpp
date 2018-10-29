
#include<iostream>
#include<cstring>
#include<cstdio>
#include<map>
#include<vector>
#include<string>
#include<ctime>
#include<cmath>
#include<cstdlib>
#include <typeinfo>
// sshpass -p "an18392886383ybj" scp xin-zh@pine44:/cl/work/xin-zh/py3env/workplace/shell/push/a.txt ./

using namespace std;

// std::vector<std::string> get_directories(const std::string& s)
// {
//     std::vector<std::string> r;
//     for(auto& p : std::filesystem::recursive_directory_iterator(s))
//         if(p.status().type() == std::filesystem::file_type::directory)
//             r.push_back(p.path().string());
//     return r;
// }

// int main(int argc,char**argv)
// {
//     std::string folder = "../../data/";
//     std::vector subfolders = get_directories(folder);
//     cout<<sunfolders<<endl;
// }

#include <dirent.h>
#include <stdio.h>
#include <string.h>

std::vector<std::string> main()
{
    const char* PATH = "../../data/";
    DIR *dir = opendir(PATH);
    std::vector<std::string> folderName;
    struct dirent *entry = readdir(dir);
    while (entry != NULL)
    {
        if (entry->d_type == DT_DIR){
            if(strlen(entry->d_name) > 3){
                std::string path(PATH);
                folderName.push_back(path+entry->d_name);
            }
        }
        entry = readdir(dir);
    }
    for (std::vector<std::string>::const_iterator i = folderName.begin(); i != folderName.end(); ++i){
        std::cout << *i << '\n';
    }
    closedir(dir);
    return folderName;
}