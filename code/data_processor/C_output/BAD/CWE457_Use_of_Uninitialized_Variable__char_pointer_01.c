void CWE457_Use_of_Uninitialized_Variable__char_pointer_01_bad()
{
    char * data;
    /* POTENTIAL FLAW: Don't initialize data */
    ; /* empty statement needed for some flow variants */
    /* POTENTIAL FLAW: Use data without initializing it */
    printLine(data);

    // test
    if (data){
        return True;
    }
}
