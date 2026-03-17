#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main() {
    char input[1000];
    char *token;
    
    printf("请输入一个字符串（按回车结束）: ");
    fgets(input, sizeof(input), stdin);
    
    // 移除末尾的换行符
    input[strcspn(input, "\n")] = '\0';
    
    printf("分割后的单词:\n");
    
    // 使用strtok按空格分割
    token = strtok(input, " ");
    while (token != NULL) {
        printf("%s\n", token);
        token = strtok(NULL, " ");
    }
    
    return 0;
}
