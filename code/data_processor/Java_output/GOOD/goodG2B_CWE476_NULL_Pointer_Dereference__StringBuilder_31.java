    private void goodG2B() throws Throwable
    {
        StringBuilder dataCopy;
        {
            StringBuilder data;

            /* FIX: hardcode data to non-null */
            data = new StringBuilder();

            dataCopy = data;
        }
        {
            StringBuilder data = dataCopy;

            /* POTENTIAL FLAW: null dereference will occur if data is null */
            IO.writeLine("" + data.length());

        }
    }
