    public void bad() throws Throwable
    {
        StringBuilder data;

        /* We need to have one source outside of a for loop in order
         * to prevent the Java compiler from generating an error because
         * data is uninitialized
         */

        /* POTENTIAL FLAW: data is null */
        data = null;

        for (int j = 0; j < 1; j++)
        {
            /* POTENTIAL FLAW: null dereference will occur if data is null */
            IO.writeLine("" + data.length());
        }
    }
