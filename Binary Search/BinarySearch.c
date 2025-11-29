#include<stdio.h>


void binarysearch(item){
    int arraye[20] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    int low = 0;
    // change it later to get the array length dynamically

    int high = sizeof(arraye)/sizeof(arraye[5]) - 1;
    // the sizeof returns the size in bytes, so depending on the data type used in the array
    // so we have the overall size in bytes 
    //and one element of the array size in bytes
    // overall/ one element gives us the number of elements in the array



    int mid;
    

    while (low <= high){


        mid = (low + high) / 2;

        if (arraye[mid] == item){
            printf("answer found, its %d \n and its memory address is %p \n", arraye[mid], &arraye[mid]);
            return;

        } else if (arraye[mid] > item){
            high = mid - 1;
            printf("guess(%d) was too high \n", arraye[mid]);
        } else if (arraye[mid] < item) {
            low = mid  + 1;
            printf("guess(%d) was too low \n", arraye[mid]);
        }

    }
    if (arraye[mid] != item){
    printf("Your Guess answer was out of list");
    }


}

int main (){




    int answer;

    printf("Enter your Guess Answer(1-20): \n");

    scanf("%d: ", &answer);
    binarysearch(answer);








    return 0;
}